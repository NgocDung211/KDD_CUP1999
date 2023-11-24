import os
from tkinter import ttk, OptionMenu,StringVar
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
import csv
from intrusionDetection import training
from functools import partial
from time import perf_counter




class App(ttk.Window):
    def __init__(self):
        super().__init__()

        self.geometry("1350x600")
        self.title("Nhóm 15- Intrusion Detection")
        self.resizable(False,False)
        self.tk.call('tk','scaling',1.3)

        self.creTabView()
    
    def creTabView(self):
        self.tabView = ttk.Notebook(self)

        #tạo frame cho tab
        self.frame = ttk.Frame(self.tabView)

        #Thêm tab vào notebook
        self.tabView.add(self.frame, text="Intrusion Detection")

        #hiển thị notebook
        self.tabView.pack(expand=True, fill="both")

        #tạo cột lớn FrameLabel
        ttk.LabelFrame(self.frame, text="", width=900, height=710).place(x=200,y=30)



        ttk.Label(self.frame, text="CHOSE DATA SET", font="Arial 15 bold", foreground="#333333").place(x=450,y=50)
        ttk.Label(self.frame, text="Data.csv:", font="Arial 12 bold", foreground="#333333").place(x=370, y=90)

        ttk.Button(self.frame, text="Data train", style="warning", width=9,
                   padding=8, cursor="hand2", command=self.open_file_train).place(x=460,y=90)
        ttk.Button(self.frame, text="Data test", style="warning", width=9,
                   padding=8, cursor="hand2", command=self.open_file_test).place(x=550,y=90)
        options = [
            "RandomForestClassifier",
            "LogisticRegression",
            "MLPClassifier",
            "DecisionTreeClassifier",
            

        ]
        ck = StringVar()
        ck.set("MLPClassifier")

        dp = OptionMenu(self.frame, ck, *options)
        dp.config(height=2,width=25,cursor="hand2")
        dp.place(x=805,y=90)
        # Set the initial selected_model
        self.selected_model = ck.get()
        def update_selected_model():
            self.selected_model = ck.get()
            print(self.selected_model)

        select_button = ttk.Button(self.frame, text="Select",style="warning", width=9,
                   padding=8, cursor="hand2",command=update_selected_model)
        select_button.place(x=700, y=90)

        


        # Bind the selection event to the function

        self.textbox1 = ScrolledText(self.frame, width=80, height=8, font="Arial 10 bold")
        self.textbox1.place(x=350, y=150)

        ttk.Button(self.frame, text="Training", style="info", width=30,
                   padding=10, cursor="hand2", command=self.train).place(x=530,y=300)
        
        ttk.Label(self.frame, text="Độ chính xác:", font="Arial 12 bold", foreground="#333333").place(x=450,y=355)
        self.textbox2 = ttk.Text(self.frame, width=80, height=8, font="Arial 10 bold")
        self.textbox2.place(x=350,y=390)


    def open_file_train(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv") ])
        if file_path.endswith(".csv"):
            self.display_csv_file_content(file_path)
        self.train_path_file = file_path
        
        return file_path
    def open_file_test(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv") ])
 
        self.test_path_file = file_path
 


    def display_csv_file_content(self, file_path):
        self.textbox1.delete("1.0", "end")
        
        with open(file_path, 'r', newline="") as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            header_text = ", ".join(header)
            self.textbox1.insert("end", "This is header of file \n\n\n")
            self.textbox1.insert("end", header_text)

    def train(self):
        t1 = perf_counter()
        print("Loading ....")
        self.textbox2.delete("1.0", "end")
        self.textbox2.insert("1.0", "Loading....\n")
        print(self.selected_model)

        sensitivity, specificity, y_test, predictions = training(self.selected_model, self.train_path_file, self.test_path_file)
        correct_count = (y_test == predictions).sum()
        incorrect_count = (y_test != predictions).sum()
        t2 = perf_counter()
        self.textbox2.insert("end", f"This is result of: {self.selected_model} model \n\n")
        self.textbox2.insert("end", f"Correct: {correct_count}\n")
        self.textbox2.insert("end", f"Incorrect: {incorrect_count}\n")
        self.textbox2.insert("end", f"True Positive Rate: {100 * sensitivity:.2f}%\n")
        self.textbox2.insert("end", f"False Negative Rate: {100 * (1- specificity):.2f}%\n")
        self.textbox2.insert("end", f"Time: {t2-t1:.2f}s\n")
        print(f"The time to train and test: {t2-t1}\n")
        
if __name__ == "__main__":
    app = App()
    app.mainloop()

