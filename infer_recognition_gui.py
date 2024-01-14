import argparse
import functools
import subprocess
import threading
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox, ttk

import numpy as np
import soundcard as sc

from mvector.predict import MVectorPredictor
from mvector.utils.record import RecordAudio
from mvector.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs', str, 'configs/cam++.yml', '配置文件')
add_arg('use_gpu', bool, False, '是否使用GPU预测')
add_arg('audio_db_path', str, 'audio_db/', '音频库的路径')
add_arg('model_path', str, 'models/CAMPPlus_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)


class VoiceRecognitionGUI:
    def __init__(self, master):
        master.title("智慧声音识别锁")
        master.geometry('950x300')
        self.master = master
        # 识别使用时间，单位秒
        self.infer_time = 2
        # 录音采样率
        self.samplerate = 16000
        # 录音块大小
        self.numframes = 1024
        # 模型输入长度
        self.infer_len = int(self.samplerate * self.infer_time / self.numframes)
        self.recognizing = False
        self.record_data = []
        self.record_audio = RecordAudio()
        # 录音长度标签和输入框
        self.record_seconds_label = tk.Label(master, text="录音长度(s):")
        self.record_seconds_label.place(x=3, y=3)
        self.record_seconds = tk.StringVar(value='3')
        self.record_seconds_entry = tk.Entry(master, width=30, textvariable=self.record_seconds)
        self.record_seconds_entry.place(x=90, y=3)
        # 判断是否为同一个人的阈值标签和输入框
        self.threshold_label = tk.Label(master, text="判断阈值:")
        self.threshold_label.place(x=4, y=40)
        self.threshold = tk.StringVar(value='0.5')
        self.threshold_entry = tk.Entry(master, width=30, textvariable=self.threshold)
        self.threshold_entry.place(x=90, y=40)
        # 选择功能标签和按钮
        self.label = tk.Label(master, text="请选择功能：")
        self.label.place(x=12, y=90)
        self.register_button = tk.Button(master, text="注册音频到声纹库", command=self.register)
        self.register_button.place(x=90, y=90)
        self.recognize_button = tk.Button(master, text="执行声纹识别", command=self.recognize)
        self.recognize_button.place(x=200, y=90)
        self.remove_user_button = tk.Button(master, text="删除用户", command=self.remove_user)
        self.remove_user_button.place(x=290, y=90)
        self.recognize_real_button = tk.Button(master, text="实时识别", command=self.recognize_thread)
        self.recognize_real_button.place(x=360, y=90)
        # 识别器
        self.predictor = MVectorPredictor(configs=args.configs,
                                          threshold=float(self.threshold.get()),
                                          audio_db_path=args.audio_db_path,
                                          model_path=args.model_path,
                                          use_gpu=args.use_gpu,
                                          )

        # 使用字典存储声音和文件信息
        self.voice_files = {}

        # 创建 ttk 样式
        self.style = ttk.Style()

        # 设置第一个按钮的样式为蓝色
        self.style.configure('Blue.TButton', foreground='blue', background='blue', padding=10,
                             font=('Microsoft YaHei', 12, 'bold'))

        # 设置第二个按钮的样式为绿色
        self.style.configure('Green.TButton', foreground='green', background='green', padding=10,
                             font=('Microsoft YaHei', 12, 'bold'))

        # 文件加锁按钮（蓝色）
        self.lock_button = ttk.Button(master, text="文件加锁", command=self.lock_file, style='Blue.TButton')
        self.lock_button.place(x=90, y=130)

        # 打开文件按钮（绿色）
        self.unlock_button = ttk.Button(master, text="打开文件", command=self.unlock_file, style='Green.TButton')
        self.unlock_button.place(x=230, y=130)

        # 用于存储选择的名称的实例变量
        self.selected_name = None

        self.result_label = tk.Label(master, text="结果展示", font=('Arial', 15))
        self.result_label.place(x=200, y=200, anchor=tk.CENTER)

        # 新增在右侧的框架和滚动条
        self.right_frame = ttk.Frame(master, borderwidth=2, relief="ridge")
        self.right_frame.place(x=450, y=0, width=500, height=300)

        self.tree = ttk.Treeview(self.right_frame, columns=("用户名称", "文件信息"))
        self.tree.column('#0', width=0, stretch=False)
        self.tree.heading("#1", text="用户名称")
        self.tree.heading("#2", text="文件信息")

        self.scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # 将声音和文件关系添加到 TreeView 中（示例数据）
        # self.predictor.users_name
        # 创建一个标签样式，用于在单元格中显示完整的文本
        # self.tree.tag_configure("details_tag", wrap=tk.WORD)
        for name in self.predictor.users_name:
            # print("===" + name + "====")
            self.tree.insert("", "end", values=(name, ""), tags=(name,))

        # 布局 TreeView 和滚动条
        self.tree.pack(side="left", expand=True, fill="both")
        self.scrollbar.pack(side="right", fill="y")

    def lock_file(self):
        self.result_label.config(text="结果展示")
        file_path = filedialog.askopenfilename(title="选择文件", filetypes=[("所有文件", "*.*")])

        if file_path:
            # Do something with the selected file path, e.g., display it or pass it to your functions
            print("Lock file:", file_path)

            self.select_name()
            # 获取用户输入的注册名称
            # name = simpledialog.askstring(title="注册", prompt="请输入注册名称")

            if self.selected_name is not None and self.selected_name != '':
                self.tree.insert(self.tree.tag_has(self.selected_name)[0], "end",
                                 values=(self.selected_name, file_path))

                self.tree.tag_configure(self.selected_name, font=("Arial", 12, "italic"),
                                        background='green')

                self.lock_voice_file(self.selected_name, file_path)

    def unlock_file(self):
        file_path = filedialog.askopenfilename(title="选择文件", filetypes=[("所有文件", "*.*")])

        if file_path:
            # Do something with the selected file path, e.g., display it or pass it to your functions
            print("Unlock file:", file_path)
            threshold = float(self.threshold.get())
            record_seconds = int(self.record_seconds.get())
            # 开始录音
            self.result_label.config(text="正在录音...")
            audio_data = self.record_audio.record(record_seconds=record_seconds)
            self.result_label.config(text="录音结束")
            name, score = self.predictor.recognition(audio_data, threshold, sample_rate=self.record_audio.sample_rate)
            if name:
                self.result_label.config(text=f"说话人为：{name}，得分：{score}")
                if file_path in self.get_files_by_voice(name):
                    try:
                        result = subprocess.run(['start', '', file_path], shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        messagebox.showinfo("提示", f"无法打开文件: {str(e)},该文件正在被占用")
                else:
                    messagebox.showerror("错误", "声纹不匹配，文件打开失败！")

            else:
                self.result_label.config(text="没有识别到说话人，可能是没注册。")
                messagebox.showerror("错误", "声纹不匹配，文件打开失败！")

    def select_name(self):
        # 创建一个新的顶级窗口
        top_window = tk.Toplevel(self.master)

        # 设置窗口大小
        top_window.geometry('300x130')

        # 添加下拉选择框到新的顶级窗口
        name_combobox = ttk.Combobox(top_window, values=self.predictor.users_name, style='TCombobox', width=20)
        name_combobox.set("请选择用户...")
        name_combobox.pack(pady=20)

        # 添加确定按钮，点击时获取选择的名称并关闭窗口
        confirm_button = tk.Button(top_window, text="确定",
                                   command=lambda: self.get_and_close(name_combobox, top_window))
        confirm_button.pack(pady=10)

        # 将顶级窗口置于最前面
        top_window.attributes('-topmost', True)
        top_window.lift()

        # 使用 wait_window 等待窗口关闭
        top_window.wait_window()

    def get_and_close(self, name_combobox, top_window):
        # 获取选择的名称
        self.selected_name = name_combobox.get()

        # 检查用户是否进行了有效选择
        if self.selected_name:
            print(f"选择的名称是: {self.selected_name}")

        # 关闭顶级窗口
        top_window.destroy()

    def lock_voice_file(self, name, file_info):
        # 添加一个声音对应多个文件的映射关系
        self.voice_files.setdefault(name, []).append(file_info)
        self.result_label.config(text=name + "的声纹锁已建立！")

    def get_files_by_voice(self, name):
        # 根据声音获取对应的文件信息列表
        return self.voice_files.get(name, [])

    # 注册
    def register(self):
        record_seconds = int(self.record_seconds.get())
        # 开始录音
        self.result_label.config(text="正在录音...")
        audio_data = self.record_audio.record(record_seconds=record_seconds)
        self.result_label.config(text="录音结束")
        name = simpledialog.askstring(title="注册", prompt="请输入注册名称")
        if name is not None and name != '' :
            if name not in self.predictor.users_name:
                self.predictor.register(user_name=name, audio_data=audio_data, sample_rate=self.record_audio.sample_rate)
                self.tree.insert("", "end", values=(name, ""), tags=(name,))
                self.result_label.config(text=name+"注册成功")
            else:
                messagebox.showinfo("提示", f"注册失败，该用户已存在")
        else:
            messagebox.showinfo("提示", f"注册失败，用户名必须不为空")

    # 识别
    def recognize(self):
        threshold = float(self.threshold.get())
        record_seconds = int(self.record_seconds.get())
        # 开始录音
        self.result_label.config(text="正在录音...")
        audio_data = self.record_audio.record(record_seconds=record_seconds)
        self.result_label.config(text="录音结束")
        name, score = self.predictor.recognition(audio_data, threshold, sample_rate=self.record_audio.sample_rate)
        if name:
            self.result_label.config(text=f"说话人为：{name}，得分：{score}")
        else:
            self.result_label.config(text="没有识别到说话人，可能是没注册。")

    def remove_user(self):
        name = simpledialog.askstring(title="删除用户", prompt="请输入删除用户名称")
        if name is not None and name != '':
            result = self.predictor.remove_user(user_name=name)
            if result:
                self.result_label.config(text="删除成功")
                # 在 TreeView 中删除所选项
                for item in self.tree.get_children():
                    values = self.tree.item(item, "values")
                    if values and values[0] == name:
                        self.tree.delete(item)

            else:
                self.result_label.config(text="删除失败")

    def recognize_thread(self):
        if not self.recognizing:
            self.recognizing = True
            self.recognize_real_button.config(text="结束声纹识别")
            threading.Thread(target=self.recognize_real).start()
            threading.Thread(target=self.record_real).start()
        else:
            self.recognizing = False
            self.recognize_real_button.config(text="实时声纹识别")

    # 识别
    def recognize_real(self):
        threshold = float(self.threshold.get())
        while self.recognizing:
            if len(self.record_data) < self.infer_len: continue
            # 截取最新的音频数据
            seg_data = self.record_data[-self.infer_len:]
            audio_data = np.concatenate(seg_data)
            # 删除旧的音频数据
            del self.record_data[:len(self.record_data) - self.infer_len]
            name, score = self.predictor.recognition(audio_data, threshold, sample_rate=self.record_audio.sample_rate)
            if name:
                self.result_label.config(text=f"【{name}】正在说话")
            else:
                self.result_label.config(text="请说话")

    def record_real(self):
        self.record_data = []
        default_mic = sc.default_microphone()
        with default_mic.recorder(samplerate=self.samplerate, channels=1) as mic:
            while self.recognizing:
                data = mic.record(numframes=self.numframes)
                self.record_data.append(data)


if __name__ == '__main__':
    root = tk.Tk()
    gui = VoiceRecognitionGUI(root)
    root.mainloop()
