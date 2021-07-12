import tkinter.filedialog as filAedialog
import tkinter as tk
import glob,sys,os
from unsupervised_learning_tools import FindNum
from numpy import array, argsort

def get_path(filename):
    return os.path.realpath(filename)

def GetFiles(in_dir,in_type,max_num):
    fs=array(glob.glob(in_dir+'/'+'*'+in_type))
    labs=[]
    for i, f_i in enumerate(fs):
        inc=i
        num=FindNum(f_i,in_type)
        labs.append(int(num))
    labs=array(labs)
    i_l=argsort(labs)[:]
    if max_num<=len(i_l):
        return fs[i_l[:max_num]], labs[i_l[:max_num]]
    else:
        return fs[i_l[:]], labs[i_l[:]]



def GetDirectories():
    master = tk.Tk()
    master.title('MATI')
    
    def input():
        global inp
        input_path = tk.filedialog.askdirectory()
        input_entry.delete(1, tk.END)  # Remove current text in entry
        input_entry.insert(0, input_path)  # Insert the 'path'
        inp=input_path
    
    def output():
        global outp
        path = tk.filedialog.askdirectory()
        output_entry.delete(1, tk.END)  # Remove current text in entry
        output_entry.insert(0, path)  # Insert the 'path'
        outp=path

    top_frame = tk.Frame(master)
    bottom_frame = tk.Frame(master)
    line = tk.Frame(master, height=1, width=400, bg="grey80", relief='groove')

    input_path = tk.Label(top_frame, text="Input Directory:")
    input_entry = tk.Entry(top_frame, text="", width=40)
    browse1 = tk.Button(top_frame, text="Browse", command=input)
    input_dir=input_entry.get()

    output_path = tk.Label(bottom_frame, text="Output Directory:")
    output_entry = tk.Entry(bottom_frame, text="", width=40)
    browse2 = tk.Button(bottom_frame, text="Browse", command=output)
    output_dir=output_entry.get()

    top_frame.pack(side=tk.TOP)
    line.pack(pady=10)
    bottom_frame.pack(side=tk.BOTTOM)

    input_path.pack(pady=5)
    input_entry.pack(pady=5)
    browse1.pack(pady=5)

    output_path.pack(pady=5)
    output_entry.pack(pady=5)
    browse2.pack(pady=5)

    l1 = tk.Label(master,  text='Select One', width=10 )
    #l1.grid(row=2,column=1)
    choices = [\
        'PCA Analysis',\
        'WL Analysis',\
        'SEA Analysis',\
        'Correlations',\
        'MachineLearning',\
        'ConstructAtomicConfiguration',\
        'ConstructSimulation',\
    ]
    variables = tk.StringVar(master)
    variables.set(choices[0])
    global opts
    opts=[]
    def my_show(value):
        global opt
        print(value)
        opt=value
        opts.append(value)
    def my_show2(value):
        global opt2
        print(value)
        opt2=value
        opts.append(value)
    opt=choices[0]
    om1 =tk.OptionMenu(master, variables, *choices,command=my_show)
    #b1 = tk.Button(master,  text='Show Value', command=lambda: my_show() )
    om1.pack(pady=5, fill=tk.X)
    #b1.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 

    choices2 = ['.dat', '.txt','.pdf', '.png', '.tiff','.jpg']
    variables2 = tk.StringVar(master)
    variables2.set(choices2[0])
    opt2=choices2[0]
    om2 =tk.OptionMenu(master, variables2, *choices2,command=my_show2)
    #b2 = tk.Button(master,  text='Show Value', command=lambda: my_show2() )
    #om1.grid(row=2,column=2)
    om2.pack(pady=5, fill=tk.X)
    #b2.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 
    def quitt():
        master.quit()    
    begin_button = tk.Button(bottom_frame, text='Done',command=quitt)
    begin_button.pack(pady=20, fill=tk.X)

    master.mainloop()
    print(inp)
    print(outp)
    print(opts[0])
    print(opts[1])
    try:
        return get_path(inp),get_path(outp),opts[0],opts[1]
    except:
        return get_path('.'),get_path('.'),opts[0],opts[1]


def GetOptionsConstrConfig():
    master = tk.Tk()
    master.title('ConstructAtomicConfiguration')
    l1 = tk.Label(master,  text='Select One', width=10 )
    #l1.grid(row=2,column=1)
    choices = [\
        'SEM-EDS Configuration',\
        'Dislocation Configuration'\
    ]
    variables = tk.StringVar(master)
    variables.set(choices[0])
    top_frame = tk.Frame(master)
    bottom_frame = tk.Frame(master)
    line = tk.Frame(master, height=1, width=400, bg="grey80", relief='groove')
             
    global opts
    opts=[]
    def my_show(value):
        global opt
        print(value)
        opt=value
        opts.append(value)
    def my_show2(value):
        global opt2
        print(value)
        opt2=value
        opts.append(value)
    opt=choices[0]
    om1 =tk.OptionMenu(master, variables, *choices,command=my_show)
    #b1 = tk.Button(master,  text='Show Value', command=lambda: my_show() )
    om1.pack(pady=5, fill=tk.X)
    #b1.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 

    choices2 = ['.dat', '.txt','.pdf', '.png', '.tiff','.jpg']
    variables2 = tk.StringVar(master)
    variables2.set(choices2[0])
    opt2=choices2[0]
    om2 =tk.OptionMenu(master, variables2, *choices2,command=my_show2)
    #b2 = tk.Button(master,  text='Show Value', command=lambda: my_show2() )
    #om1.grid(row=2,column=2)
    om2.pack(pady=5, fill=tk.X)
    #b2.pack(pady=5, fill=tk.X)
    #b1.grid(row=2,column=3) 
    def quitt():
        master.quit()    
    begin_button = tk.Button(master, text='Done',command=quitt)
    begin_button.pack(pady=5, fill=tk.X)

    master.mainloop()
    return opts[0],opts[1]
