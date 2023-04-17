from pathlib import Path
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import os
root = tk.Tk()
root.geometry("")



after_id = None
cur_img = 0
paused = True

image_folder = Path('C:\\Dataset\\imagenet')

slides = [ImageTk.PhotoImage(Image.open(filename))
            for filename in image_folder.glob('*.jpeg')]

def slide_show():
    """Change to next image (wraps around)."""
    global after_id, cur_img, paused

    if not paused:
        # Get the class name of the current image
        class_name = os.listdir(image_folder)[cur_img].split('\\')[-1].split('_')[0]

        # Check if the class name of the current image is the same as the next image
        next_class_name = os.listdir(image_folder)[(cur_img+1)%len(slides)].split('\\')[-1].split('_')[0]
        if class_name != next_class_name:
            paused = True
        else:
            cur_img = (cur_img+1) % len(slides)
            lbl.config(image=slides[cur_img])

    after_id = root.after(500, slide_show)

def start():
    global after_id, cur_img, paused

    paused = False
    if after_id:  # Already started?
        root.after_cancel(after_id)
        after_id = None
    cur_img = (cur_img+1) % len(slides)
    lbl.config(image=slides[cur_img])
    slide_show()


if len(slides) > 0:
    lbl = tk.Label(image=slides[cur_img])
    lbl.pack()

    btn_frame = tk.Frame(root)
    btn_frame.pack(side='bottom')

    btn_1 = tk.Button(btn_frame, text="start", command=start)
    btn_1.pack(side='left')

    # btn_2 = tk.Button(btn_frame, text="pause", command=pause)
    # btn_2.pack(side='left')

root.mainloop()

