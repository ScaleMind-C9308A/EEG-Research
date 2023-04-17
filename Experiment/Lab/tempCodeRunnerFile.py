self.slides = [ImageTk.PhotoImage(Image.open(filename))
            for filename in self.image_folder.glob('*.jpeg')]