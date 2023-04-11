import tkinter as tk
import os
from pathlib import Path
from PIL import Image, ImageTk
from itertools import cycle



class Application(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("Slideshow")
        self.geometry("+0+0")
        self.resizable(width=True, height=True)
        self.current_slide = tk.Label(self)
        self.current_slide.pack()
        self.duration_ms = 500

    def center(self):
        """Center the slide window on the screen"""
        self.update_idletasks()
        w = self.winfo_screenwidth()
        h = self.winfo_screenheight()
        size = tuple(int(_) for _ in self.geometry().split('+')[0].split('x'))
        x = w / 2 - size[0] / 2
        y = h / 2 - size[1] / 2
        self.geometry("+%d+%d" % (x, y))
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
    application.start()
    application.mainloop()


if __name__ == "__main__":
    import sys
    sys.exit(main())