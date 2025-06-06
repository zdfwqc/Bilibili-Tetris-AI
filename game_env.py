import cv2
import numpy as np
import pytesseract
import win32api
import win32gui
import win32con
import keyboard
import time
import pyautogui
import random

class GameEnvironment:
    def __init__(self, window_title = '欢乐俄罗斯方块'):
        self.window_title = window_title
        self.actions = ['w', 'a', 's', 'd', 'j']
        self.action_space = len(self.actions)
        self.window_handle = None
        self.find_window()
        self.score = 0
        self.done = False
        
    def find_window(self):
        """查找目标窗口"""
        self.window_handle = win32gui.FindWindow(None, self.window_title)
        if not self.window_handle:
            raise Exception(f"找不到窗口: {self.window_title}")
            
    def get_state(self):
        """获取当前状态（从OBS虚拟摄像头读取）"""
        cap = cv2.VideoCapture(1)  # 使用虚拟摄像头
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise Exception("无法读取摄像头画面")
        
        cv2.imwrite(f"images/frame.png", frame)
        # 预处理图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"images/gray.png", gray)
        # 裁剪固定区域
        x, y, w, h = 287, 425, 150, 27  # 根据实际需要调整裁剪区域
        cropped = gray[y:y+h, x:x+w]

        # 对裁剪区域进行OCR识别
        text = pytesseract.image_to_string(cropped, config='--psm 9 digits')
        try:
            current_score = int(text.strip())
            self.score = current_score
            print('\033[91m分数:', current_score, '\033[0m')  # 使用ANSI转义序列打印红色文字
        except:
            # 遮挡 说明游戏失败了
            print('\033[91m游戏失败\033[0m')  # 使用ANSI转义序列打印红色文字
            self.score = 0
            self.done = True
            pass
        cv2.imwrite("images/cropped.png", cropped)
        # 打印灰度图像
        resized = cv2.resize(gray, (480,640))  # DQN常用的输入尺寸
        normalized = resized / 255.0
        return normalized
               
    def step(self, action_idx):
        """执行动作并返回下一个状态、奖励和是否结束"""
        if not 0 <= action_idx < len(self.actions):
            raise ValueError("无效的动作索引")
            
        # 激活窗口
        win32gui.SetForegroundWindow(self.window_handle)
        time.sleep(0.1)
        
        # 发送按键
        keyboard.press_and_release(self.actions[action_idx])
        print("Action:", self.actions[action_idx])
        # 获取新状态和奖励
        time.sleep(0.4)
        curScore = self.score
        next_state = self.get_state()
        reward = self.score - curScore    
        if self.done:
            reward = -100
        return next_state, reward, self.done
        
    def reset(self):
        # 激活窗口
        # 检查窗口是否成功激活
        win32gui.SetForegroundWindow(self.window_handle)
        active_window = win32gui.GetForegroundWindow()
        if active_window != self.window_handle:
            
            win32gui.SendMessage(self.window_handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)  
            active_window = win32gui.GetForegroundWindow()
        time.sleep(0.1)
        left, top, right, bottom = win32gui.GetWindowRect(self.window_handle)
        print('find window ',(left, top, right, bottom))    
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        # 拒绝对战邀请
        win32api.SetCursorPos((center_x-110, center_y+20))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, center_x, center_y, 0, 0)
        time.sleep(random.random()*0.5)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, center_x, center_y, 0, 0)
        time.sleep(random.random()*2)

        # 重新开始游戏 点两次以防万一
        win32api.SetCursorPos((center_x-300, center_y+200))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, center_x, center_y, 0, 0)
        time.sleep(random.random()*0.5)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, center_x, center_y, 0, 0)
        time.sleep(random.random()*0.5)
        time.sleep(0.3)
        win32api.SetCursorPos((center_x-300, center_y+200))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, center_x, center_y, 0, 0)
        time.sleep(random.random()*0.5)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, center_x, center_y, 0, 0)
        time.sleep(random.random()*0.5)
        time.sleep(1)
        self.score = 0
        self.done = False
        # 这里可以添加重置游戏的具体逻辑
        return self.get_state() 
    
if __name__ == "__main__":
    env = GameEnvironment()
    env.reset()

