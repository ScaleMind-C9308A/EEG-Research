import tkinter as tk
import os
from pathlib import Path
from PIL import Image, ImageTk
from itertools import cycle
import tkinter.font as font



class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Slideshow")
        self.geometry("+0+0")
        self.resizable(width=True, height=True)
        self.duration_ms = 500

        self.columnconfigure(0, weight=1, minsize=75)
        self.rowconfigure([0, 1], weight=1, minsize=50)

        frame_img = tk.Frame(master=self, bg="black")
        frame_img.grid(row=0, column=0)
        frame_btn = tk.Frame(self)
        frame_btn.grid(row=1, column=0)
        btn = tk.Button(frame_btn, height = 2, width = 10, text="Start", command=self.start)
        btn['font'] = font.Font(size=12)
        btn.pack()
        self.current_slide = tk.Label(master=frame_img)

        self.current_slide.pack()
        # frame_img.pack(fill=tk.BOTH, expand=True)
        # frame_btn.pack()
    def center(self):
        """Center the slide window on the screen"""
        self.update_idletasks()
        # get the screen dimension
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        # window_width = 800
        # window_height = 500
        # # find the center point
        # center_x = int(screen_width/2 - window_width / 2)
        # center_y = int(screen_height/2 - window_height / 2)

        # set the position of the window to the center of the screen
        self.geometry(f'{screen_width}x{screen_height}+0+20')
    def get_jpeg_files(self, directory):
        jpeg_files = []
        for filename in os.listdir(directory):
            if filename.endswith('.JPEG'):
                jpeg_files.append(os.path.join(directory, filename))
        return jpeg_files
    
    def set_image_directory(self, path):        
        image_paths = self.get_jpeg_files(path)
        print(f"Total images found:")
        print(len(image_paths))
        self.images = cycle(zip(image_paths, map(ImageTk.PhotoImage, map(Image.open, image_paths))))
        # for image in self.images:
        #     print("New image found")

    def display_next_slide(self):
        try:
            name, self.next_image = next(self.images)
            self.current_slide.config(image=self.next_image)
            self.title(name)
            self.center()
            self.after(self.duration_ms, self.display_next_slide)
        except StopIteration:
            print("End of displaying images")
            # End of images reached, stop the slideshow
            pass

    def start(self):
        self.display_next_slide()        
    


def main():
    pwd = os.getcwd()
    imagenet_dir = os.path.abspath(os.path.join(pwd, os.pardir, os.pardir, 'Dataset/imagenet'))
    print(imagenet_dir)
    application = Application()
    application.set_image_directory(imagenet_dir)
    # application.start()
    application.mainloop()


if __name__ == "__main__":
    import sys
    sys.exit(main())