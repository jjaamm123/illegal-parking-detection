import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import csv
import datetime
import os
from roi  import ROIManager
from main import run_detection, get_first_frame

BG_BASE    = "#1e1e2e"
BG_SURFACE = "#181825"
BG_MANTLE  = "#11111b"
BG_OVERLAY = "#313244"
FG_TEXT    = "#cdd6f4"
FG_SUBTLE  = "#a6adc8"
FG_MUTED   = "#6c7086"
ACCENT_BLUE   = "#89b4fa"
ACCENT_GREEN  = "#a6e3a1"
ACCENT_RED    = "#f38ba8"
ACCENT_YELLOW = "#f9e2af"

class Dashboard:

    def __init__(self, root):
        self.root = root
        self.root.title("Illegal Parking Detection System")
        self.root.geometry("1280x760")            
        self.root.minsize(1100, 650)               
        self.root.configure(bg=BG_BASE)            

        self.video_path       = None
        self.zones            = []
        self.stop_flag        = [False]
        self.detection_thread = None              
        self.violations_log   = []

        self._build_ui()

    def _build_ui(self):
        self._build_title_bar()                  
        content = tk.Frame(self.root, bg=BG_BASE)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self._build_left_panel(content)           
        self._build_center_panel(content)
        self._build_right_panel(content)

    def _build_title_bar(self):                   
        bar = tk.Frame(self.root, bg=BG_MANTLE, height=52)
        bar.pack(fill=tk.X)
        tk.Label(bar, text="  Illegal Parking Detection System",
                 font=("Helvetica", 15, "bold"), bg=BG_MANTLE, fg=FG_TEXT
                 ).pack(side=tk.LEFT, pady=12)
        self.status_label = tk.Label(bar, text="● Idle",
                                     font=("Helvetica", 10), bg=BG_MANTLE, fg=FG_MUTED)
        self.status_label.pack(side=tk.RIGHT, padx=18)

    def _build_left_panel(self, parent):          
        panel = tk.Frame(parent, bg=BG_SURFACE, width=230)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=5)
        panel.pack_propagate(False)

        self._section(panel, "  Video Input")
        self._btn(panel, "Upload Video", self._upload_video, bg=BG_OVERLAY, fg=FG_TEXT)
        self.video_name_label = self._info_label(panel, "No video selected")

        self._section(panel, "  No-Parking Zones")
        self._btn(panel, "Define Zones on Frame", self._define_zones, bg=BG_OVERLAY, fg=FG_TEXT)
        self.zone_info_label = self._info_label(panel, "0 zones defined")

        self._section(panel, "  Time Threshold (seconds)")
        self.threshold_var = tk.IntVar(value=30)
        ttk.Scale(panel, from_=5, to=120, orient=tk.HORIZONTAL,
                  variable=self.threshold_var,
                  command=self._on_threshold_change).pack(fill=tk.X, padx=12, pady=(2, 0))
        self.threshold_val_label = tk.Label(panel, text="30 s", bg=BG_SURFACE,
                                            fg=ACCENT_GREEN, font=("Helvetica", 10, "bold"))
        self.threshold_val_label.pack(anchor=tk.W, padx=14)

        self._section(panel, "  Controls")
        self.start_btn = tk.Button(panel, text="▶  Start Detection",
                                   command=self._start_detection, relief=tk.FLAT,
                                   cursor="hand2", bg=ACCENT_GREEN, fg=BG_MANTLE,
                                   font=("Helvetica", 10, "bold"), pady=7)
        self.start_btn.pack(fill=tk.X, padx=12, pady=(3, 2))

        self.stop_btn = tk.Button(panel, text="■  Stop Detection",
                                  command=self._stop_detection, relief=tk.FLAT,
                                  cursor="hand2", bg=ACCENT_RED, fg=BG_MANTLE,
                                  font=("Helvetica", 10, "bold"), pady=7,
                                  state=tk.DISABLED)                   # ADDED: disabled by default
        self.stop_btn.pack(fill=tk.X, padx=12, pady=2)

        self._section(panel, "  Export")
        self._btn(panel, "Save Report as CSV", self._save_report, bg=BG_OVERLAY, fg=FG_TEXT)

        tk.Frame(panel, bg=BG_SURFACE).pack(fill=tk.BOTH, expand=True)
        tk.Label(panel, text="Digital Image Processing Project\nSmart Transportation",
                 font=("Helvetica", 7), bg=BG_SURFACE, fg=FG_MUTED,
                 justify=tk.CENTER).pack(pady=8)

    def _build_center_panel(self, parent):       
        panel = tk.Frame(parent, bg=BG_BASE)
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(panel, text="Live Video Feed", font=("Helvetica", 10, "bold"),
                 bg=BG_BASE, fg=FG_SUBTLE).pack(anchor=tk.W, pady=(5, 3))
        self.video_label = tk.Label(panel, bg=BG_MANTLE,
                                    text="No video loaded.\n\nUpload a video file to begin.",
                                    fg=FG_MUTED, font=("Helvetica", 12))
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def _build_right_panel(self, parent):        
        panel = tk.Frame(parent, bg=BG_SURFACE, width=290)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0), pady=5)
        panel.pack_propagate(False)

        self._section(panel, "  Statistics")
        stats_box = tk.Frame(panel, bg=BG_OVERLAY)
        stats_box.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.stat_detected   = self._stat_row(stats_box, "Vehicles Detected", "0", ACCENT_BLUE)
        self.stat_violations = self._stat_row(stats_box, "Total Violations",  "0", ACCENT_RED)
        self.stat_zones      = self._stat_row(stats_box, "Active Zones",      "0", ACCENT_YELLOW)

        self._section(panel, "  Violation Log")
        tree_frame = tk.Frame(panel, bg=BG_SURFACE)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        cols = ("ID", "Type", "Zone", "Time", "Duration")
        self.log_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=25)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background=BG_OVERLAY, foreground=FG_TEXT,
                        fieldbackground=BG_OVERLAY, rowheight=23, font=("Helvetica", 8))
        style.configure("Treeview.Heading", background=BG_MANTLE, foreground=FG_SUBTLE,
                        font=("Helvetica", 8, "bold"))
        style.map("Treeview", background=[("selected", "#585b70")])

        col_widths = {"ID": 36, "Type": 68, "Zone": 52, "Time": 62, "Duration": 58}
        for col in cols:
            self.log_tree.heading(col, text=col)
            self.log_tree.column(col, width=col_widths[col], anchor=tk.CENTER)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=sb.set)
        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=("Helvetica", 9, "bold"),
                 bg=BG_SURFACE, fg=ACCENT_BLUE, anchor=tk.W).pack(fill=tk.X, padx=8, pady=(12, 3))

    def _btn(self, parent, text, command, bg=BG_OVERLAY, fg=FG_TEXT):
        tk.Button(parent, text=text, command=command, relief=tk.FLAT, cursor="hand2",
                  bg=bg, fg=fg, font=("Helvetica", 9), pady=6,
                  activebackground="#45475a").pack(fill=tk.X, padx=12, pady=2)

    def _info_label(self, parent, text):
        lbl = tk.Label(parent, text=text, bg=BG_SURFACE, fg=FG_MUTED,
                       font=("Helvetica", 8), wraplength=200, justify=tk.LEFT)
        lbl.pack(anchor=tk.W, padx=14, pady=(0, 2))
        return lbl

    def _stat_row(self, parent, label, value, value_color):
        row = tk.Frame(parent, bg=BG_OVERLAY)
        row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row, text=label, bg=BG_OVERLAY, fg=FG_SUBTLE,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=6)
        val = tk.Label(row, text=value, bg=BG_OVERLAY, fg=value_color,
                       font=("Helvetica", 9, "bold"))
        val.pack(side=tk.RIGHT, padx=6)
        return val

    def _on_threshold_change(self, val):
        self.threshold_val_label.config(text=f"{int(float(val))} s")

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select CCTV Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.video_path = path
        self.video_name_label.config(text=os.path.basename(path), fg=FG_TEXT)
        frame = get_first_frame(path)
        if frame is not None:
            self._display_frame(frame)
        else:
            messagebox.showerror("Error", "Could not read the video file.")

    def _define_zones(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please upload a video first.")
            return
        frame = get_first_frame(self.video_path)
        roi_manager = ROIManager()
        self.zones  = roi_manager.define_zones_interactive(frame)
        count = len(self.zones)
        self.zone_info_label.config(text=f"{count} zone(s) defined",
                                    fg=ACCENT_GREEN if count > 0 else FG_MUTED)
        self.stat_zones.config(text=str(count))

    def _start_detection(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please upload a video first.")
            return
        if not self.zones:
            messagebox.showwarning("No Zones", "Please define at least one no-parking zone.")
            return

        self.violations_log = []
        for row in self.log_tree.get_children():
            self.log_tree.delete(row)
        self.stat_detected.config(text="0")
        self.stat_violations.config(text="0")

        self.stop_flag = [False]
        self.start_btn.config(state=tk.DISABLED)  
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("● Running", ACCENT_GREEN)

        output_path = os.path.splitext(self.video_path)[0] + "_output.mp4" 

        self.detection_thread = threading.Thread(
            target=run_detection,
            kwargs=dict(
                video_path     = self.video_path,
                zones          = self.zones,
                threshold      = self.threshold_var.get(),
                output_path    = output_path,
                frame_callback = self._on_frame,
                log_callback   = self._on_violation,
                stop_flag      = self.stop_flag,
                stats_callback = self._on_stats,              
            ),
            daemon=True,
        )
        self.detection_thread.start()
        self.root.after(300, self._poll_thread)              

    def _stop_detection(self):
        self.stop_flag[0] = True
        self._set_status("● Stopped", ACCENT_RED)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _save_report(self):                                   
        if not self.violations_log:
            messagebox.showinfo("No Data", "No violations have been recorded yet.")
            return
        default_name = f"parking_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")],
                                            initialfile=default_name)
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["vehicle_id", "vehicle_type",
                                                    "zone", "timestamp", "duration"])
            writer.writeheader()
            writer.writerows(self.violations_log)
        messagebox.showinfo("Saved", f"Report saved to:\n{path}")

    def _on_frame(self, frame):
        self.root.after(0, lambda f=frame: self._display_frame(f))

    def _on_violation(self, viol_info):
        self.violations_log.append(viol_info)
        self.root.after(0, lambda v=viol_info: self._add_log_row(v))

    def _on_stats(self, stats):                                
        self.root.after(0, lambda s=stats: self._update_stats(s))

    def _display_frame(self, frame):
        try:
            lbl_w = self.video_label.winfo_width()  or 800
            lbl_h = self.video_label.winfo_height() or 520
            h, w  = frame.shape[:2]
            scale = min(lbl_w / w, lbl_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(frame, (new_w, new_h))
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            imgtk   = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_label.config(image=imgtk, text="")
            self.video_label.image = imgtk
        except Exception:
            pass

    def _add_log_row(self, v):                                
        self.log_tree.insert("", 0,values=(v["vehicle_id"], v["vehicle_type"].capitalize(),v["zone"], v["timestamp"], f"{v['duration']:.0f}s"),tags=("violation",))
        self.log_tree.tag_configure("violation", foreground=ACCENT_RED)

    def _update_stats(self, stats):
        self.stat_detected.config(text=str(stats["total_detected"]))
        self.stat_violations.config(text=str(stats["total_violations"]))
        self.stat_zones.config(text=str(stats["active_zones"]))

    def _poll_thread(self):                                   
        if self.detection_thread and self.detection_thread.is_alive():
            self.root.after(500, self._poll_thread)
        else:
            if not self.stop_flag[0]:
                self._set_status("● Finished", ACCENT_BLUE)
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)


def launch_dashboard():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()
    