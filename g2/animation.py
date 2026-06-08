# animation.py 【非阻塞帧驱动版 — 不 sleep 不卡摄像头】
import tkinter as tk
import time


class DoorAnimation:
    def __init__(self):
        self.root = None
        self.canvas = None
        # 帧驱动状态
        self.state = "idle"       # idle | opening_glow | opening_keep | fail
        self.state_time = 0.0
        self.name = ""
        # 几何常量
        self.CX, self.CY, self.R = 150, 185, 85

    def start(self):
        self.root = tk.Tk()
        self.root.title("face_gate门禁")
        self.root.geometry("300x430+100+100")
        self.root.attributes('-topmost', True)
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, bg="#f0f0f0", width=300, height=430)
        self.canvas.pack(fill='both', expand=True)
        self._draw_idle()

    # ═══════════ 基础绘图 ═══════════

    def _draw_idle(self, text="门已关闭\n检测中..."):
        self.canvas.delete("all")
        self._draw_circle("#b0b0b0", "#d5d5d5", 4)
        self._draw_arrow("#b0b0b0")
        self._draw_text(text, "#999999")

    def _draw_circle(self, outline, fill, width):
        self.canvas.create_oval(
            self.CX - self.R, self.CY - self.R,
            self.CX + self.R, self.CY + self.R,
            outline=outline, fill=fill, width=width,
        )

    def _draw_arrow(self, color, scale=1.0):
        cx, cy = self.CX, self.CY
        s = scale
        pts = [
            (cx, cy - 38 * s),
            (cx + 20 * s, cy - 12 * s),
            (cx + 8 * s, cy - 12 * s),
            (cx + 8 * s, cy + 28 * s),
            (cx - 8 * s, cy + 28 * s),
            (cx - 8 * s, cy - 12 * s),
            (cx - 20 * s, cy - 12 * s),
        ]
        flat = [v for pt in pts for v in pt]
        self.canvas.create_polygon(*flat, fill=color, outline=color, width=1)

    def _draw_text(self, text, color):
        self.canvas.create_text(150, 370, text=text,
                                font=("微软雅黑", 14, "bold"), fill=color, justify="center")

    # ═══════════ 对外接口（均为瞬间状态切换，不阻塞） ═══════════

    def draw_open_door(self, name):
        """绿色开门动画（最高优先级，覆盖任何状态）"""
        self.state = "opening_glow"
        self.state_time = time.time()
        self.name = name

    def show_fail(self):
        """红色闪烁（绿色动画进行中不执行，防冲突）"""
        if self.state in ("opening_glow", "opening_keep", "fail"):
            return
        self.state = "fail"
        self.state_time = time.time()

    def restore_idle(self):
        self.state = "idle"
        self._draw_idle()

    def update(self):
        """每帧由主循环调用 —— 0 sleep，纯时间驱动"""
        if not self.root or self.state == "idle":
            return

        now = time.time()
        elapsed = now - self.state_time

        if self.state == "opening_glow":
            if elapsed >= 1.0:
                self.state = "opening_keep"
                self.state_time = now
                self._draw_green_keep()
            else:
                self._draw_green_glow(elapsed / 1.0)

        elif self.state == "opening_keep":
            if elapsed >= 2.0:
                self.state = "idle"
                self._draw_idle()
            else:
                self._draw_green_keep()

        elif self.state == "fail":
            total_flash = 3 * (0.30 + 0.25)          # 1.65s 闪烁期
            if elapsed >= total_flash + 1.5:          # 结束 → 空闲
                self.state = "idle"
                self._draw_idle()
            elif elapsed >= total_flash:               # 闪烁后保持红色
                self._draw_fail_red("陌生人\n检测失败")
            else:
                cycle = elapsed % 0.55
                if cycle < 0.30:                       # 亮红
                    self._draw_fail_red("陌生人\n检测失败")
                else:                                  # 熄灭
                    self._draw_idle("陌生人\n检测失败")

        self.root.update()

    # ═══════════ 动画帧绘制 ═══════════

    def _draw_green_glow(self, progress):
        """绿色光晕脉冲（progress: 0→1）"""
        cx, cy, r = self.CX, self.CY, self.R
        self.canvas.delete("all")

        glow_r = r + 8 + int(20 * progress)
        for layer in range(3):
            lr = glow_r - layer * 5
            green = int(46 + (200 - 46) * (1 - progress))
            self.canvas.create_oval(
                cx - lr, cy - lr, cx + lr, cy + lr,
                outline="", fill=f"#{green//2:02x}{green:02x}{green//3:02x}",
            )

        self._draw_circle("#00b400", "#00d200", 6)
        self._draw_arrow("white", 1.0 + 0.03 * progress)
        self._draw_text(f"{self.name}\n识别通过，请进！", "#2ecc71")

    def _draw_green_keep(self):
        """绿色稳态"""
        cx, cy, r = self.CX, self.CY, self.R
        self.canvas.delete("all")
        self.canvas.create_oval(
            cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 10,
            outline="", fill="#1a5c1a",
        )
        self._draw_circle("#00c800", "#00dc00", 8)
        self._draw_arrow("white")
        self._draw_text(f"{self.name}\n识别通过，请进！", "#2ecc71")

    def _draw_fail_red(self, text):
        """红色闪烁帧"""
        cx, cy, r = self.CX, self.CY, self.R
        self.canvas.delete("all")
        self.canvas.create_oval(
            cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8,
            outline="", fill="#5c1a1a",
        )
        self._draw_circle("#e74c3c", "#e74c3c", 6)
        self._draw_arrow("white")
        self._draw_text(text, "#e74c3c")

    def destroy(self):
        try:
            self.root.destroy()
        except:
            pass


# 全局实例 + 对外接口（与 detect_fixed.py 完全匹配）
door_inst = DoorAnimation()
door_start     = door_inst.start
door_open      = door_inst.draw_open_door
door_show_fail = door_inst.show_fail
door_update    = door_inst.update
destroy_all    = door_inst.destroy
