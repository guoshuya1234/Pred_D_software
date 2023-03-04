Code written by: Shuya Guo

General overview
Pred_D is an interactive desktop application for predicting diffusion coefficient of gas molecules in MOF or other porous crystal materials.The core of its computation is our trained LGBM model. The following is the source code for the interface design.

## Import the required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor as lgb
import tkinter as tk 
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image,ImageTk
import webbrowser
import joblib
import sys
import os

## Load the trained LGBM model
model = joblib.load("model/lgbm.pt") ## The detailed code for model training is in “LGBM_code.py”

## The layout design of the main interface
root= tk.Tk()
root.title("Predict diffusivity  of material on LGBM")
root.resizable(True,True) ## The window size can be changed
canvas1 = tk.Canvas(root, width = 820, height =730) ## Main window size
canvas1.pack()

## The plate layout within the main interface
re1=canvas1.create_rectangle(50,20,770,450)
re2=canvas1.create_rectangle(50,470,770,670)
re3=canvas1.create_rectangle(70,100,380,320,outline='darkgray')
re4=canvas1.create_rectangle(400,100,750,320,outline='darkgray')
re4=canvas1.create_rectangle(70,340,750,430,outline='darkgray')
re4=canvas1.create_rectangle(70,560,750,650,outline='darkgray')
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 340, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 560, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',9),text='Author：Shuya Guo,Zhiwei Qiao,Guangzhou University')
canvas1.create_window(580, 715, window=label_B)

## Message box (Related literature on molecular physical properties)
def cmx1():
    window = tk.Tk()     
    window.title('Warm prompt')     
    window.geometry('350x250')
    link = tk.Label(window, text='The physical properties of gases \nare known from the literature:\nhttps://doi.org/10.1039/B802426J'
                    , font=('microsoft yahei',10),anchor="center")
    link.place(x=30, y=50) 
    def open_url(event):
        webbrowser.open("https://doi.org/10.1039/B802426J", new=0)         
    link.bind("<Button-1>", open_url)    
btn1=tk.Button(root, text='Tooltip',font=('microsoft yahei',10), command=cmx1)
canvas1.create_window(450, 140, window=btn1)

##Message box (Instructions for Prediction of a single material diffusiivity)
def resize(w, h, w_box, h_box, pil_image): 
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height),Image.ANTIALIAS)
       
w_box = 600  
h_box = 450    

global tk_image 
photo1 = Image.open("Img/full_name.png")
w, h = photo1.size       
photo1_resized = resize(w, h, w_box, h_box, photo1)    
tk_image1 = ImageTk.PhotoImage(photo1_resized)

def cmx2():
    top2=tk.Toplevel() 
    top2.title('Instructions for use') 
    top2.geometry('620x500') 
    lab_1 = ttk.Label(top2,image=tk_image1) 
    lab_1.place(x=25, y=10) 
    top2.mainloop()  
  
btn2=tk.Button(root, text='README',font=('microsoft yahei',10), command=cmx2)
canvas1.create_window(120, 60, window=btn2)

## Message box (Instructions for batch Prediction of material diffusiivity)
def resize(w, h, w_box, h_box, pil_image):
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height), Image.ANTIALIAS)      
w_box = 600  
h_box = 500    

global tk_image 
photo2 = Image.open("Img/sample_file.png")  
w, h = photo2.size     
photo2_resized = resize(w, h, w_box, h_box, photo2)    
tk_image2 = ImageTk.PhotoImage(photo2_resized)

def cmx3():
    top1=tk.Toplevel()
    top1.title('Instructions for use')     
    top1.geometry('680x580')
    lab2 = tk.Label(top1, text='You need to create the data you want to compute\nin the format below (For example：the prediction \nof DC2H6):'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab2.place(x=20, y=20) 
    lab3 = tk.Label(top1, text='After creating the file, you can click the import file \nbutton on the screen.The predicted result  will be \nsaved in "Result/Batch_Predicted_D.xlsx".'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab3.place(x=30, y=450)
    lab3 = ttk.Label(top1,text="photo:",image=tk_image2)
    lab3.place(x=30, y=120) 
    top1.mainloop()     
btn3=tk.Button(root, text='README',font=('microsoft yahei',10),command=cmx3)
canvas1.create_window(120, 520, window=btn3)

## Sets the label and entry for entering the nine descriptor 
label_Z = tk.Label(root,font=('microsoft yahei',13),text='Predict diffusivity of material')
canvas1.create_window(415, 20, window=label_Z)

label_L = tk.Label(root,font=('microsoft yahei',11),text='Physical property of material')
canvas1.create_window(225, 100, window=label_L)

label1 = tk.Label(root,font=('microsoft yahei',10),text='HVF：') ## create 1st label box 
canvas1.create_window(160, 140, window=label1)
entry1 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 1st entry box 
canvas1.create_window(300, 140, window=entry1)

label2 = tk.Label(root,font=('microsoft yahei',10), text='PLD (Å): ') ## create 2st label box 
canvas1.create_window(160, 180, window=label2)
entry2 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 2nd entry box
canvas1.create_window(300, 180, window=entry2)

label3 = tk.Label(root,font=('microsoft yahei',10), text='LCD (Å): ') ## create 3st label box 
canvas1.create_window(160, 220, window=label3)
entry3 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 3nd entry box
canvas1.create_window(300, 220, window=entry3)

label4 = tk.Label(root,font=('microsoft yahei',10,"italic"), text='ρ') ## create 4st label box 
canvas1.create_window(110, 260, window=label4)
l0 = tk.Label(root,font=('microsoft yahei',10), text='(kg/cm^3): ') 
canvas1.create_window(170, 260, window=l0)
entry4 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 4nd entry box
canvas1.create_window(300, 260, window=entry4)

label5 = tk.Label(root, font=('microsoft yahei',10),text='VSA (m^2/cm^3): ') ## create 5st label box 
canvas1.create_window(160, 300, window=label5) 
entry5 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 5nd entry box
canvas1.create_window(300, 300, window=entry5)

label_R = tk.Label(root,font=('microsoft yahei',11),text='Physical property of gas molecules') 
canvas1.create_window(575, 100, window=label_R)

label6 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dia') ## create 1st 6abel box 
canvas1.create_window(490,180, window=label6)
l1 = tk.Label(root, font=('microsoft yahei',10),text='(Å):')
canvas1.create_window(525,180, window=l1) 
entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 6nd entry box
canvas1.create_window(660,180, window=entry6)

label7 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Pol') ## create 7st label box 
canvas1.create_window(430,220, window=label7)
l2 = tk.Label(root, font=('microsoft yahei',10,),text='(×10^25 cm^3): ')
canvas1.create_window(520,220, window=l2)
entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 7nd entry box
canvas1.create_window(660,220, window=entry7)

label8 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dip') ## create 8st label box 
canvas1.create_window(430,260, window=label8) 
l3 = tk.Label(root, font=('microsoft yahei',10),text='(×10^18 cm): ')
canvas1.create_window(510,260, window=l3)  
entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 8nd entry box
canvas1.create_window(660, 260, window=entry8)

label9 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Qua') ## create 9st label box 
canvas1.create_window(430, 300, window=label9)
l4 = tk.Label(root, font=('microsoft yahei',10),text='(×10^26 cm^2): ')
canvas1.create_window(525, 300, window=l4) 
entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 9nd entry box
canvas1.create_window(660,300, window=entry9)

## The linkage between the physical properties of molecules and molecules is realized
## Sets four properties: 1-dia,2-pol,3-Dip,4-Qua
input1 = 0
input2 = 0
input3 = 0
input4 = 0
def run2():
    dic1 = {0: 'Dia', 1: 'Pol', 2: 'Dip',3: 'Qua'}
    c2 = dic1[cm1.current()]
    if (c2 == 'Dia'):                      
        entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input1,justify='center')
        canvas1.create_window(660,180, window=entry6)              
    elif (c2 =='Pol'):
        entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input2,justify='center') 
        canvas1.create_window(660,220, window=entry7)
    elif (c2 =='Dip'):
        entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input3,justify='center') 
        canvas1.create_window(660, 260, window=entry8)                   
    elif (c2 =='Qua'):
        entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(660,300, window=entry9)
 
def calc(event):
    global input1
    global input2
    global input3
    global input4    
    dic = {0: 'He', 1: 'H2', 2: 'CO2',3: 'O2',4: 'H2S',5: 'N2',6: 'Kr',7: 'CH4',8: 'Xe',9:'C2H6'}
    c = dic[cm1.current()] 
    if (c == 'He'): 
        input1 = 2.6
        input2 = 2.04956
        input3 = 0
        input4 = 0
        entry6.delete(0,"end")
        entry6.insert(0,"2.6")
        entry7.delete(0,"end")
        entry7.insert(0,"2.04956")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"0")
    elif (c == 'H2'):
        input1 = 2.89
        input2 = 8.042
        input3 = 0
        input4 = 0.662
        entry6.delete(0,"end")
        entry6.insert(0,"2.89")
        entry7.delete(0,"end")
        entry7.insert(0,"8.042")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"0.662")
    elif (c == 'CO2'):
        input1 = 3.3
        input2 = 29.11
        input3 = 0
        input4 = 4.30
        entry6.delete(0,"end")
        entry6.insert(0,"3.3")
        entry7.delete(0,"end")
        entry7.insert(0,"29.11")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"4.30")
    elif (c == 'O2'):
        input1 = 3.46
        input2 = 15.812
        input3 = 0
        input4 = 0.39
        entry6.delete(0,"end")
        entry6.insert(0,"3.46")
        entry7.delete(0,"end")
        entry7.insert(0,"15.812")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"0.39")
    elif (c == 'H2S'):
        input1 = 3.62
        input2 = 37.82
        input3 = 0.97833
        input4 = 0
        entry6.delete(0,"end")
        entry6.insert(0,"3.62")
        entry7.delete(0,"end")
        entry7.insert(0,"37.82")
        entry8.delete(0,"end")
        entry8.insert(0,"0.97833")
        entry9.delete(0,"end")
        entry9.insert(0,"0")  
    elif (c == 'N2'):
        input1 = 3.64
        input2 = 17.403
        input3 = 0
        input4 = 1.52
        entry6.delete(0,"end")
        entry6.insert(0,"3.64")
        entry7.delete(0,"end")
        entry7.insert(0,"17.403")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"1.52") 
    elif (c == 'Kr'):
        input1 = 3.66
        input2 = 24.844
        input3 = 0
        input4 = 0
        entry6.delete(0,"end")
        entry6.insert(0,"3.66")
        entry7.delete(0,"end")
        entry7.insert(0,"24.844")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"0") 
    elif (c == 'CH4'):
        input1 = 3.8
        input2 = 25.93
        input3 = 0
        input4 = 0
        entry6.delete(0,"end")
        entry6.insert(0,"3.8")
        entry7.delete(0,"end")
        entry7.insert(0,"25.93")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"0") 
    elif (c == 'Xe'):
        input1 = 4.1
        input2 = 40.44
        input3 = 0
        input4 = 0
        entry6.delete(0,"end")
        entry6.insert(0,"4.1")
        entry7.delete(0,"end")
        entry7.insert(0,"40.44")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"0")   
    elif (c == 'C2H6'):
        input1 = 4.443
        input2 = 44.5
        input3 = 0
        input4 = 0.65
        entry6.delete(0,"end")
        entry6.insert(0,"4.443")
        entry7.delete(0,"end")
        entry7.insert(0,"44.5")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"0.65")  
        
## Create a Drop-down box
var1 = tk.StringVar() ## Create a variable 
cm1 = ttk.Combobox(root, textvariable=var1,font=('microsoft yahei',10)) ## Create a drop-down menu
cm1["value"] = ("He", "H2", "CO2","O2","H2S","N2","kr","CH4","Xe","C2H6") ## The contents of a drop-down menu
canvas1.create_window(620,140, window=cm1)
cm1.bind('<<ComboboxSelected>>', calc) ## Binding 'calc' events
  
## Main interface for input of nine descriptor values (a single molecule diffusivity)
def values():       
    global New_HVF #our 1st input variable    
    New_HVF = float(entry1.get()) 
    
    global New_PLD #our 2nd input variable
    New_PLD = float(entry2.get()) 
    
    global New_LCD #our 2nd input variable
    New_LCD = float(entry3.get()) 
    
    global New_Density #our 2nd input variable
    New_Density = float(entry4.get()) 
    
    global New_VSA #our 2nd input variable
    New_VSA =float(entry5.get()) 
    
    global New_Dia #our 2nd input variable
    New_Dia = float(entry6.get()) 
    
    global New_Pol #our 2nd input variable
    New_Pol = float(entry7.get()) 
    
    global New_Dip #our 2nd input variable
    New_Dip = float(entry8.get())
    
    global New_Qua #our 2nd input variable
    New_Qua = float(entry9.get())

## LGBM Algorithm (The predictions of a single molecule diffusivity)   
    lgD = model.predict([[New_HVF, New_PLD, New_LCD, New_Density, New_VSA, New_Dia,
                          New_Pol, New_Dip, New_Qua]])    
    ## D transformation            
    D = pow(10,lgD)
    D1 = float(D)
    D2=format(D1,'.2E')
    D3= D2.split('E')  
    if (D3[1])[0] == "-":
        D3= D3[0]+" x 10^"+ D3[1].lstrip('0')
    else:
        D3=D3[0]+" x 10^"+ (D3[1])[1:].lstrip('0')    
    
    ## label of the predicted result
    Prediction_result  = (D3)   
    label_Prediction = tk.Label(root, font=('microsoft yahei',12),width=30,height=2,
                                text= Prediction_result)
    canvas1.create_window(420, 380, window=label_Prediction)
 
    ## D label
    lbo1=tk.Label(root, font=('microsoft yahei',12,"italic"),
                                text='D:')
    canvas1.create_window(330, 380, window=lbo1)
    
    ## unit label
    lbo2=tk.Label(root, font=('microsoft yahei',12),
                                text='(cm^3/s)')
    canvas1.create_window(540, 380, window=lbo2) 

## button to call the 'values' command above       
button1 = tk.Button (root,font=('microsoft yahei',10), text='Predicted D',command=values) 
canvas1.create_window(130, 380, window=button1)

## Batch prediction of material diffusivity
label_Z1 = tk.Label(root,font=('microsoft yahei',12),text='Batch prediction of material diffusivity')
canvas1.create_window(415, 470, window=label_Z1)

## Open File
def open_file():
    filename = filedialog.askopenfilename(title='open exce')
    entry_filename.delete(0,"end")
    entry_filename.insert('insert', filename)
 
button_import = tk.Button(root, text="Import File",font=('microsoft yahei',10),command=open_file)
canvas1.create_window(280, 520, window=button_import)
 
## Import File
entry_filename = tk.Entry(root,font=('microsoft yahei',10),width=30)
canvas1.create_window(520, 520, window=entry_filename)

## Output LGBM model prediction results
def print_file():

    ## get extract contents of entry
    a = entry_filename.get() 

    ## Load the dataset
    pred_data1=pd.read_excel(a)
 
    ## Divide the data set
    pred_data=pred_data1.dropna(axis=0)

    ## Divide the data set
    df = pd.DataFrame(pred_data,columns=[ 'LCD', 'HVF', 'VSA', 'PLD', 'Density', 'Dia',
         'Pol', 'Dip', 'Qua ','lgD'])
    X_pred = df[['LCD', 'HVF', 'VSA', 'PLD', 'Density', 'Dia',
         'Pol', 'Dip', 'Qua ']].astype(float)

    ## Standardization
    transfer=StandardScaler()
    X_pred=transfer.fit_transform(X_pred)  
  
    ##model prediction
    Y_predict2 = model.predict(X_pred) 
    D = pow(10,Y_predict2) ## D transformation

    ## output result
    d1 = pd.DataFrame({'D_pred':D}) 
    newdata = pd.concat([pred_data,d1],axis=1) 
    newdata.to_excel("Result/Batch_Predicted_D.xlsx")
    
    ## label_P (Prediction complete)
    label_P = tk.Label(root, font=('microsoft yahei',12),
                                text='Predicted results have default stored in:\nResult/Batch_Predicted_D.xlsx', bg='green')
    canvas1.create_window(450, 600, window=label_P)    
    
## Prediction button
but_pre=tk.Button(root,font=('microsoft yahei',10)
             , text='Batch Predicted D', bg='orange', command=print_file)
canvas1.create_window(160, 600, window=but_pre)

root.mainloop() 
