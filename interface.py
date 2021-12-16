import tkinter as tk

root= tk.Tk()

canvas = tk.Canvas(root, width = 1200, height = 700,  relief = 'raised')
canvas.pack()

label = tk.Label(root, text='Hate-Speech Recognizer')
label.config(font=('helvetica', 18, 'bold'))
canvas.create_window(600, 225, window=label)

label = tk.Label(root, text='Enter sentence:')
label.config(font=('helvetica', 14, 'bold'))
canvas.create_window(600, 300, window=label)

entry = tk.Entry(root) 
canvas.create_window(600, 340, window=entry)

def getSquareRoot ():
    label = tk.Label(root, text= '',font=('helvetica', 14, 'bold'))
    canvas.create_window(600, 530, window=label)

    x = entry.get()
    
    label = tk.Label(root, text= 'The Square Root of ' + x + ' is',font=('helvetica', 14))
    canvas.create_window(600, 480, window=label)
    
    label = tk.Label(root, text= float(x)**0.5,font=('helvetica', 14, 'bold'))
    canvas.create_window(600, 530, window=label)
    
button = tk.Button(text='Run', command=getSquareRoot, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
canvas.create_window(600, 400, window=button)

root.mainloop()