from tkinter import *
from tkinter.colorchooser import askcolor


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='SAVE', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
        
        self.choose_size_button1 = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button1.grid(row=0, column=5)

        self.c = Canvas(self.root, bg='white', width=640, height=640)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        from PIL import ImageGrab
        x=self.root.winfo_rootx()+self.root.winfo_x()
        y=self.root.winfo_rooty()+self.root.winfo_y()
        x1=x+self.root.winfo_width()
        y1=y+self.root.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).save("poop.jpg")
        global guess
        guess()
        
    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

from PIL import Image

def save_as_png(canvas,fileName):
    # save postscipt image 
    canvas.postscript(file = fileName + '.eps') 
    # use PIL to convert to PNG 
    img = Image.open(fileName + '.eps') 
    img.save(fileName + '.png', 'png') 



from torch.autograd import Variable

import torch.onnx
import torchvision.transforms as transforms
from torch import nn
import torchvision
import torch
from PIL import Image
import numpy as np

IMAGE_SIZE = 64

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_SIZE*IMAGE_SIZE * 3, 3*64),
            nn.ReLU(),
            nn.Linear(3*64, 256),
            nn.ReLU(),
            nn.Linear(256, 15),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



dummy_input = Variable(torch.randn(1, 3, 64, 64))
print("starting")
model = NeuralNetwork()
model.load_state_dict(torch.load('ballhaha.pth'))
model.eval()

def guess():
    im = Image.open("poop.jpg")
    transform = transforms.Compose([transforms.Resize((64 * 3, 64 * 1)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    convert = transform(im)
    print(convert)

    result = model(convert)
    print(result[0])
    classes = {
        0: "American Football",
        1: "Baseball",
        2: "Basketball",
        3: "Billard Ball",
        4: "Bowling Ball",
        5: "Cricket Ball",
        6: "Football",
        7: "Golf Ball",
        8: "Hockey Ball",
        9: "Hockey Buck",
        10: "Rugby Ball",
        11: "Shuttlecock",
        12: "Table Tennis Ball",
        13: "Tennis Ball",
        14: "Volleyball"
        
    }
    print(classes[torch.argmax(result[0]).item()])
    print(classes[torch.argmax(result[1]).item()])
    print(classes[torch.argmax(result[2]).item()])
    return classes[torch.argmax(result[0]).item()]

    
if __name__ == '__main__':
    Paint()