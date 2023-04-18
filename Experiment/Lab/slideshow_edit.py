from pathlib import Path
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import os
import tkinter.font as font

class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Slideshow")
        self.geometry("+0+0")
        self.resizable(width=True, height=True)
        self.duration_ms = 500
        self.after_id = None
        self.cur_img = 0
        self.paused = True
        pwd = os.getcwd()
        image_folder_path = os.path.abspath(os.path.join(pwd, os.pardir, os.pardir, 'Dataset/imagenet'))
        self.image_folder = Path(image_folder_path)
        print(image_folder_path)
        print('Total image found: ')
        print(sum(1 for x in self.image_folder.glob('*.JPEG') if x.is_file()))        
        self.slides = [ImageTk.PhotoImage(Image.open(filename))
            for filename in self.image_folder.glob('*.JPEG')]
        frame_img = tk.Frame(master=self, width=200, height=200, bg="black")
        self.lbl = tk.Label(image=self.slides[self.cur_img], master = frame_img)
        self.lbl.pack()
        frame_img.pack(fill=tk.BOTH, expand=True)
       
       
        
    def center(self):
        """Center the slide window on the screen"""
        self.update_idletasks()
        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        size = tuple(int(_) for _ in self.geometry().split('+')[0].split('x'))
        x = w / 2 - size[0] / 2
        y = h / 2 - size[1] / 2
        self.geometry("+%d+%d" % (x, y))
        
    # def center(self):
    #     """Center the slide window on the screen"""
    #     self.update_idletasks()
    #     # get the screen dimension
    #     screen_width = self.winfo_screenwidth()
    #     screen_height = self.winfo_screenheight()
    #     window_width = 800
    #     window_height = 500
    #     # find the center point
    #     center_x = int(screen_width/2 - window_width / 2)
    #     center_y = int(screen_height/2 - window_height / 2)

    #     # set the position of the window to the center of the screen
    #     self.geometry(f'{screen_width}x{screen_height}+{center_x}+{center_y}')
    def slide_show(self):
        if not self.paused:
            # Get the class name of the current image
            class_name = os.listdir(self.image_folder)[self.cur_img].split('\\')[-1].split('_')[0]

            # Check if the class name of the current image is the same as the next image
            next_class_name = os.listdir(self.image_folder)[(self.cur_img+1)%len(self.slides)].split('\\')[-1].split('_')[0]
            if class_name != next_class_name:
                self.paused = True
            else:
                self.cur_img = (self.cur_img+1) % len(self.slides)
                self.lbl.config(image=self.slides[self.cur_img])
        self.after_id = self.after(self.duration_ms, self.slide_show)
    def start(self):
        self.paused = False
        if self.after_id:  # Already started?
            self.after_cancel(self.after_id)
            self.after_id = None
        self.cur_img = (self.cur_img+1) % len(self.slides)
        self.lbl.config(image=self.slides[self.cur_img])
        self.slide_show()
    def run(self):
        print('Total image found: ')
        print(sum(1 for x in self.image_folder.glob('*jpeg') if x.is_file()))
        if len(self.slides) > 0:
            btn_frame = tk.Frame(self)
            btn_frame.pack(side='bottom')

            # define font
            myFont = font.Font(size=12)

            btn_1 = tk.Button(btn_frame, height = 2, width = 10, text="Start", command=self.start)
            btn_1['font'] = myFont
            btn_1.pack(side='left')
def main():
    application = Application()
    application.run()
    application.center()
    application.mainloop()
if __name__ == "__main__":
    import sys
    sys.exit(main())


