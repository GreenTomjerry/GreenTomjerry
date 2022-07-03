#!/usr/bin/python
# -*- coding: UTF-8 -*-

# In[1]:
#注：可以通过y添加参数-d 后跟数据因变量，也可以不添加。
#本脚本运行方式为python 4.25WUSTstronger.py -i inputfile -d dependentvalue 
#注：仅输入文件是必须添加的参数，因变量可以不添加，不会报错

#未更新：将分类变量和连续变量分析相关性时，可以考虑F检验或者eta方检验。
#尚未研究出如何画其他分类变量的蜂巢图 考虑使用supplot方法手动实现蜂巢图
#核心智能分析报告尚无改进灵感，需要与梁博确认才可以
#4.25更新1.增加了画连续变量相关性的热力图功能，使用spearman相关性，而非皮尔逊相关性
#更新2：修正了数据填充的顺序，从而不会干涉最初的数据分布
#更新3：能够识别id，序号等完全无用的干扰特征，将其直接删除
#更新4：能够识别0 1，将0 1直接定义为分类变量
#更新5：现在整数和浮点数分布图都可以看到count值了
#更新6：现在可以用随机森林算法算出自变量对因变量的影响大小了

#依旧注释掉了自动生成压缩包的功能！图片将会被自动输出为pdf
#目前文件只能接受xlsx文件，目的是少出bug，想增加的话也可以随时增加
#目前遇到的问题是，本脚本会自动分类数据，并且用基于统计的方法寻找哪些特征之间有相关性，并绘图
#运行方式为python 4.25WUSTstronger.py -i inputfile -d dependentvalue 
#目前，指定输出文件夹的功能处于不可用状态，只需要 -i这一个参数
#输出文件会被自动生成于脚本所在路径下一个叫tempreport的文件夹下
#本代码极限情况下生成的文件是6个.pdf文件，以及一个记录日志的txt文件 log.txt

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
import shutil
if os.path.exists("tempreport"):
    shutil.rmtree("tempreport")
else:
    os.makedirs("tempreport")
    
###设置字体等，为输出pdf报告做准备
from reportlab.lib import colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, LongTable, Image
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
#from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
story = []
table_data = []
table_data_group = []
pdfmetrics.registerFont(TTFont('SimSun', r'C:\Windows\Fonts\simsun.ttc'))
pdfmetrics.registerFont(TTFont('simhei', r'C:\Windows\Fonts\simhei.ttf'))
stylesheet = getSampleStyleSheet()
Title = stylesheet['Title']
BodyText = stylesheet['BodyText']
Title.fontName = 'simhei'
BodyText.fontName = 'SimSun'

table_style = [
    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
    ('TEXTCOLOR', (0, 0), (2, 0), colors.red),  # 设置表格内文字颜色
    ('GRID', (0, 0), (-1, -1), 0.1, colors.black),  # 设置表格框线为黑色，线宽为0.1
]

table_style_group = [('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
                     ('GRID', (0, 0), (-1, -1), 0.1, colors.black),  # 设置表格框线为黑色，线宽为0.1
                     ]

###设置完毕

import zipfile
#打包目录为zip文件（未压缩）
def make_zip(source_dir, output_filename):
  zipf = zipfile.ZipFile(output_filename, 'w')
  pre_len = len(os.path.dirname(source_dir))
  for parent, dirnames, filenames in os.walk(source_dir):
    for filename in filenames:
      pathfile = os.path.join(parent, filename)
      arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
      zipf.write(pathfile, arcname)
  zipf.close()


import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages

#mpl.rcParams['font.family']='SimHei'
#mpl.rcParams['font.sans-serif']=['SimHei']
#mpl.rcParams['axes.unicode_minus']=False 



# 读取数据

#设置传递参数，只需要加入输入文件名（含路径与输入文件名）和输出文件路径（仅目录）

#dataname="car_crashes.xlsx"
#dataname="sndHsPr.xlsx"
#dataname="Telco-Customer-Churn.csv"
#dependentvalue="total"
#dependentvalue="Churn"
#dependentvalue="price"

import sys, getopt


opts, args = getopt.getopt(sys.argv[1:], "h:i:d:o:")

input_file=""

output_file=""

for op, value in opts:

    if op == "-i":

        dataname = value

    elif op == "-o":

        outputpath = value
        
    elif op == "-d":

        dependentvalue = value

    elif op == "-h":

        print("test.py -i <inputfile> -o <outputpath\.zip> -d dependent variable")

        sys.exit()

print(dataname)


#print(outputpath)

# In[3]:


#!wget -nc "https://labfile.oss.aliyuncs.com/courses/2616/seaborn-data.zip"
#!unzip seaborn-data.zip -d ~/
#sns.get_dataset_names()
#这个代码块的意义是下载并查看能够使用哪些示例数据集


# In[3]:

def read_set_data(dataname):
    import pandas as pd
    # 将数据导入内存，生成日志文件
    '''
    datasuffix=dataname.split(".")[-1]
    if datasuffix in ["xlsx" , "xlsm" , "xls" , "xlt"]:
       #print("cao")
       data=pd.read_excel(f"{dataname}")#从Excel文件导入数据
    elif datasuffix=="csv":
        data=pd.read_csv(f"{dataname}")#从CSV文件导入数据
    elif datasuffix=="txt":
        data=pd.read_table(f"{dataname}")#从限定分隔符的文本文件导入数据
    '''
    
    # # 一、导入数据

    # 可导入来自不同文件格式的数据集，如xlsx，csv等，只需要去除最前面的#号并且填入文件的路径，名称即可

    data=pd.read_excel(f"{dataname}")#从Excel文件导入数据
    #data=pd.read_csv(f"{dataname}")
    #data=pd.read_table("/home/hejunjie/jerry/bam/depthchr1_lower200.txt")#从限定分隔符的文本文件导入数据
    #data=pd.read_sql(query, connection_object)#从SQL表/库导入数据
    #data=pd.read_json(json_string)#从JSON格式的字符串导入数据
    #data=pd.read_html(url)#解析URL、字符串或者HTML文件，抽取其中的tables表格
    #data=pd.read_clipboard()#从你的粘贴板获取内容，并传给read_table()
    #data=pd.DataFrame(dict)#从字典对象导入数据，Key是列名，Value是数据
    #data=pd.read_excel(f"{dataname}")
    
    # 读取数据完成
    
    file_handle=open(r'tempreport\log.txt',mode='w')
    
    file_importance=open(r'tempreport\report.txt',mode='w')
    
    return file_handle,file_importance,data

        #创建运行日志文件

#自定义代码，使运行过程可以被输出到控制台，并被储存到文件中

def writing(x,filename):
    print(x)
    filename.write(str(x)+"\n")

#将文字加入列表中预备之后输出为pdf
def drawpdf(file,x,datatype):
    file.append(Paragraph(x, datatype))

#自定义代码，方便计算eta方，从而计算连续变量和分类变量之间关联
def correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta


# 出于熟悉使用等目的，可导入来自seabron库的玩具数据集

# # 二、数据分类打包

def data_classify(data):

    writing("欢迎使用本数据可视化功能，您的统计数据将会被按照数据类型分类，并可视化后输出。",file_handle)
    writing("开始统计您的特征种类和特征名：",file_handle)
    # 向文件中输入运行信息
    
    writing(f"您的统计数据共有{len(data.columns)}个特征，分别为{data.columns.to_list()}",file_handle)

    datadict=data.dtypes.to_dict()

#将数据放入字典，根据其数据类型进行分流，分别进行可视化检验

    for key, value in datadict.items():
        valuestr=str(value)
        datadict[key]=valuestr
    
#将数据种类转化为字符串，方便后续的循环判断

#这里是关键，将数据分类实现分别处理，因此如果出错，结果会比较严重
#（一些整数数字变量可能会被认为是“object”变量，如"1","2","3"......）
    stri="数据分类打包中......数据种类将被逐一报告"+"\n"
    print(stri)
    file_handle.write(stri)
    objectlist=[]
    floatlist=[]
    intlist=[]
    for key, value in datadict.items():
        if value == "object" :
            objectlist.append(key)
            writing(f"{key}是分类变量"+"\n",file_handle)
        elif value == "category":
            objectlist.append(key)
            writing(f"{key}是分类变量"+"\n",file_handle)
        elif value == "float64":
            floatlist.append(key)
            writing(f"{key}是浮点数变量"+"\n",file_handle)
        if value =="int64":
            if len(data[key].unique())==2:
                objectlist.append(key)
                writing(f"{key}只有两个值，被分为分类变量，请检查"+"\n",file_handle)
            elif len(data[key].unique())==1:#删除毫无变化的数字
                writing(f"{key}只有唯一值，不会被分析，请检查"+"\n",file_handle)
                #删除很可能是id或者编号的数字
            elif (data[key] == np.arange(start=1,stop=len(data.index)+1)).all() or (data[key] ==data.index).all():
                 writing(f"{key}极有可能是id或编号，不会被分析，请检查"+"\n",file_handle)
                 data.drop(key,axis=True,inplace=True)
            else:
                intlist.append(key)
                writing(f"{key}是整数变量"+"\n",file_handle)
#将数据根据类型分门别类打包以等待不同的可视化方法
    dataobject=data[objectlist]
    datafloat=data[floatlist]
    dataint=data[intlist]
    writing("数据分类打包完成",file_handle)
#分属不同数据类型的数据被按列切割为多个子数据集，预备之后的分别处理
    return dataobject,datafloat,dataint,[objectlist,floatlist,intlist]

# # 三、分类变量数据预处理

def True_check_NA(df):
    null_percent = (np.sum(df.isnull()/len(df)))*100
    if null_percent>50:
        writing("*****超过50%为空白数据，建议检查*****",file_handle)
    writing("缺失数值占总共百分比为"+str(round(null_percent,1))+"%",file_handle)
def check_NA_all(df):
    #判断输入是数据框还是series
    if isinstance(df, pd.DataFrame):
        for i in df.columns:
            writing("#"*50,file_handle)
            writing(i, file_handle)
            True_check_NA(df[i])
    elif isinstance(df, pd.Series):
        True_check_NA(df)

def count_state(df):
    for i in df.columns:
        writing("#"*50,file_handle)
        writing(i,file_handle)
        writing(df[i].unique(),file_handle)
        writing(df[i].value_counts(),file_handle)

def classification_data_process(dataobject,featurelist):

# # 由于可能存在日期等分类信息，故将带有日期的特征，种类过于多的特征从object类的特征中直接删除，会另外编写将日期转化为新特征的脚本代码
    
    writing("正在检测分类特征质量，如果特征中不同取值数多于20，它将被删除，并报告",file_handle)
    if len(featurelist[0])>0:
        hardfeature=[]
        
        for i in dataobject.columns:
            
            if len(dataobject[i].unique())>20:
                hardfeature.append(i)
                writing(f"特征{dataobject[i]}被删除，疑似为种类过多的特征，请检查其是否为日期或人名",file_handle)
                dataobject.drop(i,axis=1,inplace=True) 
                featurelist[0].remove(i)
        writing("正在显示筛选后的分类特征的种类及占比",file_handle)
        count_state(dataobject)
        writing("显示完成",file_handle)
        if hardfeature:
            writing(f"警告{hardfeature}等特征已经被删除，不会被可视化",file_handle)
        else:
            writing("本次检查结束，没有特征被删除",file_handle) 
        #删除过多缺失的特征
        
        writing("正在检测分类变量数据缺失度，缺失超过90%的特征将会被删除......",file_handle)
        writing("......",file_handle)
        #writing((dataobject.isnull().sum())/dataobject.shape[0]).sort_values(ascending=False).map(lambda x:"{:.2%}".format(x),file_handle)
        check_NA_all(dataobject)
        bar_dict = ((dataobject.notnull().sum())/dataobject.shape[0]).sort_values(ascending=False).to_dict()
        #使用一个字典将以上结果储存
        delete=[]
        for key,value in bar_dict.items():
            if value<0.1:
                delete.append(key)
                featurelist[0].objectlist.remove(key)
        if len(delete)==0:
            writing("所有特征都被保留",file_handle)
        else:
            dataobject.drop(delete,axis=1,inplace=True)
            writing(f"共有{len(delete)}个特征，因为数据缺失率过高而被删除，请检查数据质量，它们是{delete}",file_handle)
            
        #return dataobject

#坐标：此处为执行原本fillna的位置，但由于会破坏数据原本的分布从而被放弃

# # 四、连续变量数据预处理

# 处理float变量

#删除过多缺失的特征
def float_data_process(datafloat,featurelist):
    writing("正在检测float变量数据缺失度，缺失超过90%的特征将会被删除......",file_handle)
    writing("......",file_handle)
    writing("缺失度以百分比显示",file_handle)
    nonelist=str(((datafloat.isnull().sum())/datafloat.shape[0]).sort_values(ascending=False).map(lambda x:"{:.2%}".format(x)))
    if len(nonelist)>0:
        #writing(nonelist,file_handle)
        check_NA_all(datafloat)
    else:
        writing("没有连续变量特征，请检查",file_handle)
    bar_dict = ((datafloat.notnull().sum())/datafloat.shape[0]).sort_values(ascending=False).to_dict()
    #使用一个字典将以上结果储存
    delete=[]
    for key,value in bar_dict.items():
        if value<0.1:
            delete.append(key)
            featurelist[1].remove(key)
    if len(delete)==0:
        writing("所有特征都被保留",file_handle)
    else:
        datafloat.drop(delete,axis=1,inplace=True)
        writing(f"共有{len(delete)}个特征，因为数据缺失率过高而被删除，请检查数据质量，它们是{delete}",file_handle)

    #return datafloat



# 处理整数变量

def int_data_process(dataint,featurelist): 
    writing("正在检测int数变量数据缺失度，缺失超过90%的特征将会被删除......",file_handle)
    writing("......",file_handle)
    if len(featurelist[2])>0:  
        writing("缺失度以百分比显示",file_handle)
        #writing(str(((dataint.isnull().sum())/dataint.shape[0]).sort_values(ascending=False).map(lambda x:"{:.2%}".format(x))),file_handle)
        check_NA_all(dataint)
        bar_dict = ((dataint.notnull().sum())/dataint.shape[0]).sort_values(ascending=False).to_dict()
        #使用一个字典将以上结果储存
        delete=[]
        for key,value in bar_dict.items():
            if value<0.1:
                delete.append(key)
                featurelist[2].remove(key)
        if len(delete)==0:
            writing("所有特征都被保留",file_handle)
        else:
            dataint.drop(delete,axis=1,inplace=True)
            writing(f"共有{len(delete)}个特征，因为数据缺失率过高而被删除，请检查数据质量，它们是{delete}",file_handle)
    else:
        writing("没有整数变量，请检查",file_handle)
    
    #return dataint


# # 四、原始数据的可视化报告

# # 展示条形统计图

def draw_cla(dataobject):

    writing("正在为您绘画分类数据的柱形图",file_handle)
    #myfont=FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
    #sns.set(font=myfont.get_name())
    
    if len(dataobject.columns)>0:
        with PdfPages("tempreport\destribution_category.pdf") as pdf:
            for i in dataobject.columns:
                sns.set_style('white')
                ######################################################
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                sns.catplot(x=i, kind="count", data=dataobject,margin_titles=True)
                plt.xticks(rotation=15)
                #sns.catplot(x=i, kind="count", data=dataobject)
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        writing("绘画完成",file_handle)
    else:
        writing("您的数据中没有分类变量，请检查",file_handle)
    
# # 一些特征可能存在极端的离群值，这会导致生成很差的难以阅读的分布图，以下步骤能够对浮点数和整数（int64）数据集进行初步的数据预处理，将分位数99%以上，1%以下的数据替换为分位数99%和1%的数据，从而避免上述情况的发生，这个数值也可以调整

def draw_int(dataint):
    writing("一些特征可能存在极端的离群值，这会导致生成很差的难以阅读的分布图，以下步骤能够对浮点数和整数（int64）数据集进行初步的数据预处理，将分位数99%以上，1%以下的数据替换为分位数99%和1%的数据，从而避免上述情况的发生，这个数值也可以调整",file_handle)
    if len(dataint.columns)>0:
        writing("正在为整数特征生成分布图",file_handle)
        writing("为保证图像质量，会将每个特征中大于99%的数值替换为max，小于1%的数值替换成min",file_handle)
        for i in dataint.columns:
            highest=round(dataint[i].quantile(0.99))
            lowest=round(dataint[i].quantile(0.01))
            print(highest,lowest)
            dataint[i].clip(lower=lowest, upper=highest,inplace=True)
            writing(f"{i} max = {highest},min = {lowest}",file_handle)
            
                    
        with PdfPages("tempreport\destribution_int.pdf") as pdf:
            #myfont=FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
            #sns.set(font=myfont.get_name())
            #sns.set()  # 声明使用 Seaborn 样式
            #输出每一个列数据的分布图
            for i in dataint.columns:
                #plt.figure()
                ######################################################
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                plt.figure()
                sns.set_style('white')
                sns.displot(x=i,data=data)
                #plt.tight_layout()
                pdf.savefig()
                '''
                代码存档，该代码会导致生成核密度图
                sns.distplot(dataint[i])
                plt.tight_layout()
                pdf.savefig()
                '''
                plt.close()
                writing("绘画完成",file_handle)
    else:
          writing("您的数据中没有整数变量，请检查",file_handle)

# In[27]:

def draw_float(datafloat):
    if len(datafloat.columns)>0:
        writing("正在为浮点数特征生成分布图",file_handle)
        writing("为保证图像质量，会将每个特征中大于max的数值替换为max，小于min的数值替换成min",file_handle)     
        for i in datafloat.columns:
            highest=datafloat[i].quantile(0.99)
            lowest=datafloat[i].quantile(0.01)
            print(highest,lowest)
            datafloat[i].clip(lowest,highest,inplace=True)
     
            writing(f"{i} max = {highest},min = {lowest}",file_handle)
            
        writing("正在为浮点数特征生成分布图",file_handle)
        sns.set()  # 声明使用 Seaborn 样式
        #输出浮点数的密度分布图
        with PdfPages("tempreport\destribution_float.pdf") as pdf:
            for i in datafloat.columns:
                plt.figure()
                sns.set_style('white')
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                sns.displot(x=i,data=data)
                #plt.tight_layout()
                pdf.savefig()
                plt.close()
        writing("绘画完成",file_handle)
    else:
          writing("您的数据中没有浮点数变量，请检查",file_handle)
    
    # 以上的数据处理做到了将妨碍绘图的所有数据都预先清除的作用。但还无法做到将所有数据都完全处理，如，填充NA，标准化，去离群值，等等。这些会在后面逐一完成，下一步先将它们的关联绘图输出

# 连续数据与连续数据之间的关联图如下

def data_process(dataobject,datafloat,dataint):
#在分析数据相关性和进行机器学习前，需先将数据中的na值填充

#填充na值
    writing("数据关联性分析开始，需要对数据进行预处理",file_handle)
    writing("数据预处理进行中......分类变量的缺失值将被以众数填充",file_handle)
    for i in dataobject.columns.tolist():
        mode=dataobject[[i]].mode().iat[0, 0]
        dataobject[i].fillna(mode,inplace=True)
    writing("填充完成",file_handle)
        

    writing("数据预处理进行中......int数值的缺失值将被以中位数填充",file_handle)
    for i in dataint.columns.tolist():
        median = dataint[i].median()
        dataint[i].fillna(median,inplace=True)
    
    writing("填充完成",file_handle)
    
    
    writing("数据预处理进行中......float数值的缺失值将被以中位数填充",file_handle)
    
    for i in datafloat.columns.tolist():
        median = datafloat[i].median()
        datafloat[i].fillna(median,inplace=True)
    
    writing("填充完成",file_handle)
    
    #自此完成数据的预处理部分，fillna完成
    
    frame=[dataint,datafloat,dataobject]
    datacombine=pd.concat(frame,axis=1)
    return  dataobject,datafloat,dataint,datacombine

#计算浮点数变量间的spearman相关性
def datafloat_corr(datafloat):
    writing("正在为浮点数特征之间生成关联图",file_handle)
    
    if len(datafloat.columns)>1:
        float_corr = datafloat.corr(method='spearman')
        with PdfPages("tempreport\cor_float&float.pdf") as pdf: 
            #plt.figure()
            sns.set_style('white')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            sns.pairplot(data)
            pdf.savefig()
            plt.close()
            #创建一个mask遮盖半边热图
            plt.figure()
            mask = np.zeros_like(float_corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            # 绘制图像
            f, ax = plt.subplots(figsize=(12, 9))
            sns.heatmap(float_corr,mask =mask,  vmax=1, center=0, annot=True, fmt='.1f',
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}) #mask=mask,
            pdf.savefig()
            plt.close()
    else:
        writing("您的数据中浮点数特征数量小于2，请检查",file_handle)


# 分类变量与数字变量之间的关系图绘制（箱型图）并输出到同路径下为pdf


#用皮尔逊相关性是不够好的，这个就在下版本更新改良吧！########
#浮点数与分类变量之间
def float_cla_corr(datafloat,dataint,dataobject,featurelist):
    writing("正在为数字变量（整数与浮点数）与分类变量生成关联图",file_handle)
    if (len(datafloat.columns)>0 or len(dataint.columns)>0) and len(dataobject.columns)>0:   
        count=0
        picture_dict={}
        have_corr={}
        No_corr={}
        for i in dataint.columns:
            picture_dict[i]=[]
        for i in datafloat.columns:
            picture_dict[i]=[]
        for i in featurelist[0]:#这段代码的思路是：先找到所有分类变量，再将每一个分类变量与每一个数字变量的皮尔逊相关系数做检查，
            #如果有大于我们要求的，将其放入字典中储存
            df_dummies = pd.get_dummies(datacombine[i])
            df_new = pd.concat([datacombine, df_dummies], axis=1)
            df_new.drop(featurelist[0],axis=1,inplace=True)
            df_corr=df_new.corr()
            delete_line=df_dummies.columns.to_list()
            data_CatandFloat_corr=df_corr.loc[:,delete_line]
            data_CatandFloat_corr.drop(delete_line,axis=0,inplace=True)
            #这样就使得只判断一个分类变量中的每一项和数字变量的皮尔逊相关性
            for s in data_CatandFloat_corr.index:
                featurelist1=[]
                #print(i)
                #具体判断部分，使用lamba函数解决
                corr_obivious=[x for x in data_CatandFloat_corr.loc[s] if abs(x) > 0.4 and abs(x) !=1.0]
                corr_notobivious=[x for x in data_CatandFloat_corr.loc[s] if abs(x) < 0.4 and abs(x) !=1.0]
                if len(corr_obivious) >= 1:
                    have_corr[s]=[i,corr_obivious]
                    #writing(f"{s} 与 {i}存在关联性")
                    picture_dict[s].append(i)
                    #writing(f"卡方值为{corr_obivious}")
                    count+=1
                else:
                    No_corr[s]=[i,corr_notobivious]
                    #writing(f"{s}与{i}不存在关联性")
                    #writing(f"关联性过低，皮尔逊相关值为{corr_notobivious}")
        for key,value in have_corr.items():
            writing(f"{key} 与 {value[0]}存在关联性",file_handle)
            writing(f"皮尔逊相关值为{value[1]}",file_handle)
        for key,value in No_corr.items():
            writing(f"{key} 与 {value[0]}不存在关联性",file_handle)
            writing(f"关联性过低，皮尔逊相关值为{value[1]}",file_handle)
        delet_pic_dic=[]
        for key,value in picture_dict.items():
            if len(value)==0:
                delet_pic_dic.append(key)
        for i in delet_pic_dic:
            picture_dict.pop(i)
        writing(f"共有{count*2}张图片即将被输出,包括箱型图和直方图",file_handle)
        from matplotlib.backends.backend_pdf import PdfPages
        if count>0:
            with PdfPages("tempreport\cor_float&category_filt.pdf") as pdf: 
                for key,value in picture_dict.items():
                    '''
                    for v in value:
                        #myfont=FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
                        #sns.set(font=myfont.get_name())
                        sns.set_style('white')
                        fig=datacombine.boxplot(column=key,by=v).get_figure()
                        plt.xticks(rotation=15)
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close()
                    '''
                    for v in value:
                        #myfont=FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
                        #sns.set(font=myfont.get_name()
                        print(f"{key},{value},x={v},y={key}画图中")
                        plt.rcParams['font.sans-serif'] = ['SimHei']
                        plt.rcParams['axes.unicode_minus'] = False
                        sns.set_style('white')
                        sns.barplot(x=v,y=key,data=datacombine)
                        pdf.savefig()
                        plt.close()
            writing("绘图完成",file_handle)
    else:
        writing("浮点数特征或分类变量缺失，无法绘图，请检查",file_handle)
        

# 分类变量与分类变量之间的关系图绘制（条形图）

# 下一单元格的代码会筛选相关特征，再画出相关联分类特征两两之间的关联图
def object_corr(dataobject):
    #必须有分类变量，才能绘图
    if len(dataobject.columns)>1: 
        #import statsmodels.api as sm
        import scipy.stats as stats
        count=0
        picture_dict={}
        for i in dataobject.columns:
            picture_dict[i]=[]
        for i in dataobject.columns:
                for s in dataobject.columns:
                    if i==s:
                        continue
                    else:
                        table_sp = pd.crosstab(dataobject[i], dataobject[s])
                        p=stats.chi2_contingency(table_sp)[1]
                        if p > 0.05:
                            writing(f'不拒绝原假设,无充分证据表明{i}与{s}因素之间存在关系',file_handle)
                        else:
                            if p < 0.05:
                                writing(f'显著性水平α=0.05下,拒绝原假设,{i}与{s}充分相关,建议作图分析做进一步判断到底是因子里面哪个因素起作用',file_handle)
                                picture_dict[i].append(s)
                                count+=1
                            elif p < 0.01:
                                writing(f'显著性水平α=0.01下,拒绝原假设,{i}与{s}显著相关,建议作图详细分析',file_handle)
                                picture_dict[i].append(s)
                                count+=1
        delet_pic_dic=[]
        for key,value in picture_dict.items():
            if len(value)==0:
                delet_pic_dic.append(key)
        for i in delet_pic_dic:
            picture_dict.pop(i)
        writing(f"共有{count}张图片即将被输出",file_handle)
        if count>0:
            with PdfPages("tempreport\cor_category&category_filt.pdf") as pdf:
                for key,value in picture_dict.items():
                    for v in value:
                        #myfont=FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
                        #sns.set(font=myfont.get_name())
                        sns.set_style('white')
                        plt.rcParams['font.sans-serif'] = ['SimHei']
                        plt.rcParams['axes.unicode_minus'] = False
                        datacombine_df=pd.crosstab(dataobject[key], dataobject[v], margins=True, normalize=True)
                        datacombine_df.drop('All', inplace=True)
                        datacombine_df.drop('All', axis=1, inplace=True)
                        datacombine_df.plot(kind="bar")
                        plt.xticks(rotation=15)
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
            writing("绘画完成",file_handle)
        else:
            writing("您的数据中分类特征数量小于2，请检查",file_handle)
    
####4.25版新增功能，通过随机森林筛选特征中对于感兴趣的因变量最为重要的数个特征
def transfer_cate_tonum(dataframe):
    """
    这个函数是为了将分类变量映射为数字而编写的，将其映射为数字后此变量就可以被随机森林等
    模型接受。
    示例：
    [{'haidian': 0,
  'chaoyang': 1,
  'shijingshan': 2,
  'xicheng': 3,
  'fengtai': 4,
  'dongcheng': 5},
 {'middle': 0, 'high': 1, 'low': 2},
 {'海淀区': 0, '朝阳区': 1, '石景山区': 2, '西城区': 3, '丰台区': 4, '东门区': 5}]
    """
    diclist=[]
    objectname=[]
    count=0

    for i in dataframe.describe(include="object").columns:
        elenum=list(np.arange(len(dataframe[i].unique())))
        ele =list(dataframe[i].unique())
        #print(i)
        #print(ele,elenum)
        objectname.append(i)
        dataframe[i]=dataframe[i].map(dict(zip(ele,elenum)))
        exec("obdict%s = dict(zip(ele,elenum))"%count)
        #exec("dataframe[i]=dataframe[i].map(object%s)"%count)
        exec("diclist.append(obdict%s)"%count)
        count+=1
    return objectname,diclist
    

def logic_unit(dataframe,mostinpo,dependentvalue):
    """
    根据因变量和自变量的种类决定要使用什么函数判定其相关
    """
    if mostinpo not in featurelist[0] and dependentvalue not in featurelist[0]:
        #假如最重要自变量和因变量是整数或浮点数的话......
        judging_Monotonicity_ll(dataframe,mostinpo)
    elif mostinpo in featurelist[0] and dependentvalue not in featurelist[0]:
        #假如最重要自变量是分类变量，而因变量是浮点数的话......
        judging_Monotonicity_ol(dataframe,mostinpo)
    elif mostinpo in featurelist[0] and dependentvalue in featurelist[0]:
        #假如最重要自变量因变量都是分类变量的话.....
        judging_Monotonicity_oo(dataframe,mostinpo)
        
def judging_Monotonicity_oo(dataframe,mostinpo):
    '''
    如果自变量和因变量都是object，画堆叠条形图
    '''
    cnt = pd.crosstab(dataframe[mostinpo], dataframe[dependentvalue])    # 构建特征与目标变量的列联表
    #plt.show()    # 展示图像
    with PdfPages(r"tempreport\report_picture.pdf") as pdf:
        plt.figure()
        sns.set_style('white')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        cnt.plot.barh(stacked=True, figsize=(15,6))    # 绘制堆叠条形图，便于观察不同特征值流失的占比情况
        pdf.savefig()
        plt.close()

def judging_Monotonicity_ol(dataframe,mostinpo):
    temp = dataframe.groupby(mostinpo).mean()[dependentvalue]

    templen = len(temp)
    writing(f"以下为{mostinpo}取“0”和“1”时，{dependentvalue}的平均值", file_importance)
    drawpdf(story,f"以下为{mostinpo}取“0”和“1”时，{dependentvalue}的平均值",BodyText)
    drawpdf(story," ",BodyText)
    writing(f"{temp}", file_importance)
    for i,c in temp.items():
        table_data_group.append([i,round(c,3)])
    table_group =  Table(data=table_data_group, style=table_style, colWidths=[30,70])
    story.append(table_group)
    if temp.loc[0]<temp.loc[1]:
        writing(f"“1”对{dependentvalue}有正面影响", file_importance)
        drawpdf(story,f"“1”对{dependentvalue}有正面影响",BodyText)
        
        writing(f"“0”对{dependentvalue}有负面影响", file_importance)
        drawpdf(story,f"“0”对{dependentvalue}有负面影响",BodyText)
    elif temp.loc[0]>temp.loc[1]:
        writing(f"“1”对{dependentvalue}有负面影响", file_importance)
        drawpdf(story,f"“1”对{dependentvalue}有负面影响",BodyText)
        
        writing(f"“0”对{dependentvalue}有正面影响", file_importance)
        drawpdf(story,f"“0”对{dependentvalue}有正面影响",BodyText)
        
    with PdfPages(r"tempreport\report_picture.pdf") as pdf:
        plt.figure()
        sns.set_style('white')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        g = sns.FacetGrid(data, col=mostinpo)
        g.map(plt.hist, dependentvalue) 
        #plt.tight_layout()
        pdf.savefig()
        plt.close()
    
        

def judging_Monotonicity_ll(dataframe,mostinpo):
    """
    判断因变量和自变量之间是什么样的关联
    """
    X = dataframe.dropna()[mostinpo]
    y =  dataframe[dependentvalue]
    X_new = np.array(X).reshape(-1,1)
    print(X_new.shape,y.shape)

    from sklearn.linear_model import LinearRegression 
    
    lr = LinearRegression()
    lr.fit(X_new,y)
    from sklearn.feature_selection import f_regression
    ff,fv=f_regression(X_new,y)
    if lr.coef_>0:
        if fv<0.05:
            writing(f"P-value={fv},自变量{mostinpo}与因变量{dependentvalue}呈正比关系", file_importance)
        else:
            writing(f"P-value={fv}自变量{mostinpo}与因变量{dependentvalue}关系复杂，大致呈正比关系", file_importance)
    elif lr.coef_<0:
        if fv<0.05:
            writing(f"P-value={fv},自变量{mostinpo}与因变量{dependentvalue}呈反比关系", file_importance)
        else:
            writing(f"P-value={fv},自变量{mostinpo}与因变量{dependentvalue}关系复杂，大致呈反比关系", file_importance)

    with PdfPages(r"tempreport\report_picture.pdf") as pdf:
            plt.figure()
            sns.set_style('white')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.scatter(x=X,y=y)
            plt.xlabel(mostinpo)
            plt.ylabel(dependentvalue)
            #plt.tight_layout()
            pdf.savefig()
            plt.close()
    writing(f"已针对自变量{mostinpo}与因变量{dependentvalue}之间关系生成图，请前往report_picture.pdf查看",file_importance)
    
    
#以随机森林，将重要特征按重要性依次输出的程序
def print_inportance(X,y):
    #from sklearn.model_selection import cross_val_score
    #writing("sklearn.model_selection import cross_val_score正常",file_handle)
    from sklearn.ensemble import RandomForestRegressor
    #writing("sklearn.ensemble import RandomForestRegressor正常",file_handle)
    #from sklearn.datasets import load_iris
    #from numpy.core.umath_tests import inner1d
    import numpy as np
    #writing("np正常",file_handle)
    
    #Forest_reg = RandomForestRegressor()
    
    Forest_model = RandomForestRegressor(n_estimators=100)
    #writing("建模正常",file_handle)
    Forest_model.fit(X,y)
    #writing("拟合正常",file_handle)
    #scores = cross_val_score(Forest_model, X_test,y_test,scoring="neg_mean_squared_error")
    #mse_score = np.sqrt(-scores)
    #print((mse_score.mean(), mse_score.std()))
    importances = Forest_model.feature_importances_
    #name =Forest_model.feature_names_in_()
    print(importances) 
    
    importances = Forest_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X.columns[0:]
    count=0
    for f in range(X.shape[1]):
        if count==0:
            
            mostinpo = feat_labels[indices[f]]
            #mostinpo = "AREA"
            
            writing(f"{feat_labels[indices[f]]}对{dependentvalue}影响最大",file_importance)
            drawpdf(story,f"{feat_labels[indices[f]]}对{dependentvalue}影响最大",BodyText)
            logic_unit(data,mostinpo,dependentvalue)
            drawpdf(story,"自变量对因变量的影响如下所示，排序：降序",BodyText)
            drawpdf(story," ",BodyText)
            writing("自变量对因变量的影响如下所示，排序：降序",file_importance)
            #judging_Monotonicity(data,mostinpo)
        writing("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]),file_importance)
        table_data.append([str(f+1)+")",feat_labels[indices[f]],round(importances[indices[f]],3)])
        #drawpdf(story,"自变量对因变量的影响如下所示，排序：降序",BodyText)
        count+=1
    table = Table(data=table_data, style=table_style, colWidths=[30,100,80])
    story.append(table)

def vif_calculate(dataobject,datafloat,dataint):
    frame=[dataobject,datafloat,dataint]
    datacombine=pd.concat(frame,axis=1)
    datacombine.drop(dependentvalue,axis=1,inplace=True)
    datacombine["constant"]=1
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    for i in range(len(datacombine.columns)-1):
        VIF = variance_inflation_factor(datacombine.values, i)
        print(VIF)
        if VIF > 10:
            writing(f"请关注自变量{datacombine.columns[i]}，可能出现严重多重共线性",file_importance)

        
def RF_training(dependentvalue,data):            
    if dependentvalue in data.columns:
        writing(f"您输入了因变量{dependentvalue}作为感兴趣的变量，将以随机森林模型学习其他特征对因变量的影响",file_handle)
        writing("分类特征转码中，将会把分类特征转化为数字以适应树模型",file_handle)
        if len(featurelist[0])>0:
            objectname,diclist=transfer_cate_tonum(dataobject)
        else:
            writing("数据集没有分类特征，无需转码", file_handle)
        frame=[dataint,datafloat,dataobject]
        datacombine=pd.concat(frame,axis=1)
        X = datacombine.drop(dependentvalue,axis=1)
        y = datacombine[dependentvalue]
        writing("训练中,结果将会被单独输出于report中",file_handle)
        #writing(f"{dataname}文件机器学习简略报告",file_importance)
        drawpdf(story,f"{dataname}文件机器学习简略报告",Title)
        #writing(f"因变量：{dependentvalue}",file_importance)
        drawpdf(story,f"因变量：{dependentvalue}",BodyText)
        print_inportance(X,y)
        writing("完成，结果已输出",file_handle)
        vif_calculate(dataobject,datafloat,dataint)
            
            #from sklearn.model_selection import train_test_split
    else:
        writing(f"您输入的因变量{dependentvalue}有误，请检查",file_handle)
    
             
    writing("相关性分析结束",file_handle)  

def trans_txt_to_pdf():
    #import tempfile
    #from reportlab.lib import colors
    
    #from reportlab.lib.enums import TA_JUSTIFY
    #from io import BytesIO
    
    
    #story.append(Paragraph("sndHsPr.xlsx文件机器学习简略报告", Title))
    #story.append(Paragraph("因变量：price", BodyText))
    '''
    with open(r"tempreport\report.txt","r") as txt:
        count=0
        for line in txt.readlines():
            if count == 0:
                story.append(Paragraph(line, Title))
                count+=1
            story.append(Paragraph(line, BodyText))
    '''
    
    doc = SimpleDocTemplate(r'tempreport\report.pdf')
    doc.build(story)

#inpathname=dataname.split("\\").pop()

#zipfilepath="tempreport"    
#make_zip(zipfilepath,outputpath)
#shutil.rmtree("tempreport")

from PyPDF2 import PdfFileReader, PdfFileWriter


def concat_pdf(filename, save_filepath):
    """
    合并多个PDF文件
    @param filename:文件名
    @param read_dirpath:要合并的PDF目录
    @param save_filepath:合并后的PDF文件路径
    @return:
    """
    pdf_writer = PdfFileWriter()
    for filename in filename:
        print(filename)
        #filepath = os.path.join(read_dirpath, filename)
        # 读取文件并获取文件的页数
        pdf_reader = PdfFileReader(filename)
        pages = pdf_reader.getNumPages()
        # 逐页添加
        for page in range(pages):
            pdf_writer.addPage(pdf_reader.getPage(page))
    # 保存合并后的文件
    with open(save_filepath, "wb") as out:
        pdf_writer.write(out)
    print("文件已成功合并，保存为："+save_filepath)
    


if __name__=='__main__':
    
    file_handle,file_importance,data = read_set_data(dataname)
    #读取参数，生成初始数据框和两个报告文件
    
    dataobject,datafloat,dataint,featurelist=data_classify(data)
    #对数据集进行分类
    
    classification_data_process(dataobject,featurelist)
    #对分类变量数据集进行处理
    
    float_data_process(datafloat,featurelist)
    #对浮点数变量数据集进行处理
    
    int_data_process(dataint,featurelist)
    #对整数变量数据集进行处理
    
    draw_cla(dataobject)
    #画分类数据的条形图
    
    draw_float(datafloat)
    #画浮点数数据的分布图
    
    draw_int(dataint)
    #画整数数据的分布图
    
    dataobject,datafloat,dataint,datacombine=data_process(dataobject,datafloat,dataint)
    #对数据进行fillna等处理，之后才能画关联图
    
    datafloat_corr(datafloat)
    #浮点数之间关联图
    
    object_corr(dataobject)
    #分类特征关联图
    
    float_cla_corr(datafloat,dataint,dataobject,featurelist)
    #分类，浮点数之间关联图
    try:
        if dependentvalue:
            RF_training(dependentvalue,data)
    #随机森林机器学习
    except:
        writing("本次流程没有指定因变量，不会进行随机森林建模，分析结束",file_handle)
        writing("本次流程没有指定因变量，report文件为空",file_importance)
    
    file_handle.close()
    file_importance.close()
    
    try:
        if dependentvalue:
            trans_txt_to_pdf()
    except:
        pass
    #concat_pdf([r"tempreport\report.pdf",r"tempreport\report_picture.pdf"],r"tempreport\MLreport.pdf")