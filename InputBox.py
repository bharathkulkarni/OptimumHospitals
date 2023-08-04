####File name: InputBox.py
import tkinter
from tkinter import *
root = Tk()
e1 = Entry(root, width=55)
l1=Label(root, justify=LEFT)
result = ""
def ShowDialog(s):
	 root.title("Input Box")

	 l1.grid(row=0, column=1, sticky=W, padx= 10, pady=5)
	 l1['text'] = s

	 e1.grid(row=1, column=1, padx= 10, pady=5)

	 button1 = Button(root, text=" OK ", command = button1_Click)
	 button1.grid(row=2, column=1, sticky=E, padx= 10, pady=5)

	 root.mainloop()

def button1_Click():
	 f = open('tttemp', 'w')
	 f.write(e1.get())
	 f.close()
	 e1.delete(0, END)
	 l1['text'] = ""
	 root.quit()

def GetInput():
	 f = open('tttemp', 'r')
	 result = f.read()
	 f.close()
	 import os
	 os.remove('tttemp')
	 return result
