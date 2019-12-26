import tkinter as tk
from tkinter import messagebox
import os, os.path
import glob
from PIL import Image, ImageDraw, ImageTk


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class Application(tk.Frame):
    painter_size = 4
    item_tag = 'mask'
    # 最多标记5个对象
    painter_colors = {
        0: 'black', # eraser
        1: 'red',
        2: 'green',
        3: 'blue',
        4: 'yellow',
        5: 'purple',
    }
    current_image = None
    current_mask = None
    current_index = 0

    def __init__(self, master=None,
                 root_dir='../data/object-detect/test2'):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

        self.root_dir = root_dir

        self.images_dir = os.path.join(self.root_dir, 'images')
        ensure_dir(self.images_dir)
        self.image_files = glob.glob(os.path.join(self.images_dir, '*'))
        self.masks_dir = os.path.join(self.root_dir, 'masks')
        ensure_dir(self.masks_dir)
        self.canvas_dir = os.path.join(self.root_dir, 'canvas')
        ensure_dir(self.canvas_dir)

        self.load_state()

        self.set_image(self.current_index)
        # self.set_image(Image.open(image_files[0]))

    def load_state(self):
        state_file_path = os.path.join(self.root_dir, 'state.txt')
        if os.path.exists(state_file_path):
            with open(state_file_path, 'r') as fp:
                self.current_index = int(fp.read())

    def save_state(self):
        state_file_path = os.path.join(self.root_dir, 'state.txt')
        with open(state_file_path, 'w+') as fp:
            fp.write(str(self.current_index))

    def get_file_name(self, index):
        img_file_path = self.image_files[index]
        file_name = os.path.basename(img_file_path)
        file_name = file_name[:file_name.rindex('.')]
        return img_file_path, file_name

    def set_image(self, index):
        if index < 0 or index > len(self.image_files)-1:
            messagebox.showerror('错误', '到头了')
            return

        self.canvas.delete('all')

        img_file_path, file_name = self.get_file_name(index)

        self.master.title(f'{img_file_path} {index+1}/{len(self.image_files)}')

        self.current_image = ImageTk.PhotoImage(file=img_file_path)
        self.canvas.create_image(0, 0, image=self.current_image, anchor=tk.NW)

        canvas_file_path = os.path.join(self.canvas_dir, f'{file_name}.txt')
        if os.path.exists(canvas_file_path):
            with open(canvas_file_path, 'r') as fp:
                for l in fp.readlines():
                    x0, y0, x1, y1, id = [s for s in l.strip().split(',')]
                    id = int(id)
                    self.canvas.create_oval(
                        float(x0), float(y0), float(x1), float(y1),
                        fill=self.painter_colors[id],
                        outline='',
                        tags=[self.item_tag, id],
                    )
                    # self.canvas.create_text(x0, y0, text=id)
                    self.object_id.set(id)

        self.current_index = index
        self.save_state()

    def save_current(self):
        _, file_name = self.get_file_name(self.current_index)
        if self.current_image:
            msk_file_path = os.path.join(self.masks_dir, f'{file_name}.png')
            img = Image.new('L', (self.current_image.width(), self.current_image.height()))
            draw = ImageDraw.Draw(img)
            for i in self.canvas.find_withtag(self.item_tag):
                draw.ellipse(self.canvas.coords(i), fill=int(self.canvas.gettags(i)[1]))
            img.save(msk_file_path, 'PNG')
        canvas_file_path = os.path.join(self.canvas_dir, f'{file_name}.txt')
        with open(canvas_file_path, 'w+') as fp:
            for i in self.canvas.find_withtag(self.item_tag):
                _, id = self.canvas.gettags(i)
                x0, y0, x1, y1 = self.canvas.coords(i)
                fp.write(f'{x0},{y0},{x1},{y1},{id}{os.linesep}')

    def move_next(self):
        self.save_current()
        self.set_image(self.current_index+1)

    def move_back(self):
        self.save_current()
        self.set_image(self.current_index-1)

    def undo(self):
        if not self.canvas_commands:
            messagebox.showerror('错误', '到头了')
        else:
            id = self.canvas_commands.pop()
            self.canvas.delete(id)

    def create_widgets(self):
        self.top_bar = tk.Frame(self)
        self.top_bar.pack(side='top')

        self.previous = tk.Button(self.top_bar, text="Back",
                              command=self.move_back)
        self.previous.pack(side=tk.LEFT)

        self.next = tk.Button(self.top_bar, text="Next",
                              command=self.move_next)
        self.next.pack(side=tk.LEFT)

        self.object_id = v = tk.IntVar()
        for id, color in self.painter_colors.items():
            object_id1 = tk.Radiobutton(self.top_bar, variable=self.object_id, value=id, background=color)
            object_id1.pack(side=tk.LEFT)
        self.object_id.set(1)

        self.center = tk.Frame(self.master)
        self.center.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.center, borderwidth=3, relief=tk.SOLID)
        self.canvas.bind('<B1-Motion>', self.paint_mask)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.bind_all('<Key>', self.press_key)

    def press_key(self, event):
        if event.char is '1':
            self.painter_size = 4
        elif event.char is '2':
            self.painter_size = 8
        elif event.char is '3':
            self.painter_size = 16
        elif event.char is '4':
            self.painter_size = 32

    def paint_mask(self, event):
        id = self.object_id.get()
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasx(event.y)
        x0 = x-self.painter_size
        y0 = y-self.painter_size
        x1 = x+self.painter_size
        y1 = y+self.painter_size
        if id:
            self.canvas.create_oval(
                x0, y0, x1, y1,
                fill=self.painter_colors[id],
                outline='',
                tags=[self.item_tag, id],
            )
        else:
            cns = self.canvas.find_overlapping(x0, y0, x1, y1)
            for i in cns:
                if self.item_tag in self.canvas.gettags(i):
                    self.canvas.delete(i)


root = tk.Tk()
root.minsize(800, 600)
app = Application(master=root)
app.mainloop()