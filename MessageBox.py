#File name: MessageBox.py
import tkinter
from tkinter import *
from PIL import Image, ImageTk
def Show(stri):
	root = Tk()
	root.title('Final output')
	print(stri)
	
	photo=PhotoImage(file=stri)
	label=Label(root,image=photo)
	label.pack()
	root.mainloop()