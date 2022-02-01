from tkinter import *
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
import py2neo
from py2neo import Node, Relationship, Graph, Path, Subgraph
from py2neo import NodeMatcher, RelationshipMatcher
from gnql import *
from tkinter import ttk

import nltk


# action to connect to graph database
def connect():
    try:
        global graph
        # get parameters put by user
        url = entry_url.get()
        password = entry_password.get()
        username = entry_username.get()
        # vocabular extraction
        graph, label_node, label_relation, properties_key, value_node = Vocabulary_extractor(url, username, password)
        connection_status.delete("1.0", 'end')
        connection_status.insert("1.0", "You are connected as user neo4j to:\n URL: " + url)
        # loop to insert each property in her place
        nodelist.delete(0, END)
        for i in label_node:
            nodelist.insert(END, i)

        relationlist.delete(0, END)
        for i in label_relation:
            relationlist.insert(END, i)

        propertylist.delete(0, END)
        for i in properties_key:
            propertylist.insert(END, i)
    # exception handling
    except Exception as e:
        connection_status.delete("1.0", 'end')
        connection_status.insert("1.0", e)
        nodelist.delete(0, END)
        nodelist.insert("1.0", e)
        relationlist.delete(0, END)
        relationlist.insert("1.0", e)
        propertylist.delete(0, END)
        propertylist.insert("1.0", e)


# NL action handling
def request():
    global query
    global query_gen
    try:
        query = nldsQl(entry_query.get("1.0", END))
        query_gen.delete("1.0", 'end')
        query_gen.insert("1.0", query)
        list_log.insert(END, ">>> " + str(entry_query.get("1.0", END)))
    except Exception as e:
        query_gen.delete("1.0", 'end')
        query_gen.insert("1.0", e)
        list_log.insert(END, ">>> Error: " + str(e), "\n")


# Cypher execution
def execution():
    global response
    global accur
    try:
        query_g = query_gen.get("1.0", END)
        response = graph.run(query_g)
        mylist.delete(0, END)
        for i in response:
            mylist.insert(END, i)
        k = []
        k.append(str(list_log.get('@1,0', END)))
        k = "".join(k)

        all_log = []
        for x in k:
            log_str = re.sub(r"(?<=[a-z])r?n", " ", x)
            log_str = log_str.replace('>>> ', "")
            all_log.append(log_str)
        query_nl = "".join(query_gen.get("1.0", END))
        accur = jaccard_distance(set(ngrams(all_log, 20)), set(ngrams(query_nl, 20)))
        if accur * 100 > 98.99:
            precy = "*****/*****"
        elif accur * 100 > 98.8:
            precy = "****/*****"
        elif accur * 100 > 98.5:
            precy = "**/*****"
        elif accur * 100 < 97:
            precy = "*/*****"

        text_accuracy.delete("1.0", 'end')
        text_accuracy.insert("1.0", "Precision is:\n" + precy)
    except Exception as e:
        mylist.delete(0, END)
        mylist.insert(END, e)
        text_accuracy.delete("1.0", 'end')
        text_accuracy.insert("1.0", " ")


# NLDS windows
window = Tk()
window.title('NLDS-QL')
window.geometry("1000x700")
# fen.iconbitmap
window.config()

# Title label
title_label = Label(window, text=" Welcome on Natural Language data science questions to queries on graphs",
                    font=(" None", 20))
title_label.pack(side=TOP)
# creation of frame graph database frame
label_frame_connect = LabelFrame(window, text=" Graph data base connection", font=(14))  # main Label
label_frame_connect.place(relx=0.5, rely=0.075, relwidth=1, relheight=0.30, anchor='n')

label_url = Label(label_frame_connect, text="URL", font=(12))  # URL label
label_url.place(relx=0.02, rely=0.09, anchor='n')
label_username = Label(label_frame_connect, text="user name", font=(12))  # user_nameLabel
label_username.place(relx=0.04, rely=0.26, anchor='n')
label_password = Label(label_frame_connect, text="Password", font=(12))  # Password Label
label_password.place(relx=0.04, rely=0.5, anchor='n')

connection_status_lab = Label(label_frame_connect, text="Connection status", font=(12))  # Password Label
connection_status_lab.place(relx=0.4, rely=0.01, anchor='n')

graph_info_lab = LabelFrame(label_frame_connect, text="Label node", font=(12))  # Password Label
graph_info_lab.place(relx=0.65, rely=0.01, anchor='n')
graph_info_bar = Scrollbar(graph_info_lab, bg='blue')
graph_info_bar.pack(side=RIGHT, fill=Y)
nodelist = Listbox(graph_info_lab, yscrollcommand=graph_info_bar.set, bg='#F2EEEE')
nodelist.pack(fill="both", expand="yes")
graph_info_bar.config(command=nodelist.yview)

node_v = Scrollbar(graph_info_lab, orient=HORIZONTAL)
node_v.pack(side=BOTTOM, fill=X)
node_v.config(command=nodelist.xview)

relation_info_lab = LabelFrame(label_frame_connect, text="Label Relation", font=(12))  # Password Label
relation_info_lab.place(relx=0.8, rely=0.01, anchor='n')
relation_info_bar = Scrollbar(relation_info_lab)
relation_info_bar.pack(side=RIGHT, fill=Y)
relationlist = Listbox(relation_info_lab, yscrollcommand=relation_info_bar.set, bg='#F2EEEE')
relationlist.pack(fill="both", expand="yes")
relation_info_bar.config(command=relationlist.yview)

relation_v = Scrollbar(relation_info_lab, orient=HORIZONTAL)
relation_v.pack(side=BOTTOM, fill=X)
relation_v.config(command=relationlist.xview)

property_info_lab = LabelFrame(label_frame_connect, text="Properties", font=(12))  # Password Label
property_info_lab.place(relx=0.93, rely=0.01, anchor='n', relheight=1)

property_info_bar = Scrollbar(property_info_lab)
property_info_bar.pack(side=RIGHT, fill=Y)
propertylist = Listbox(property_info_lab, yscrollcommand=property_info_bar.set, bg='#F2EEEE')
propertylist.pack(fill="both", expand="yes")
property_info_bar.config(command=propertylist.yview)

property_v = Scrollbar(property_info_lab, orient=HORIZONTAL)
property_v.pack(side=BOTTOM, fill=X)
property_v.config(command=propertylist.xview)

connection_status = StringVar()
connection_status = Text(label_frame_connect, font=("Segoe Print", 10), foreground="red")  # Password Label
connection_status.place(relx=0.42, rely=0.15, relwidth=0.20, relheight=0.50, anchor='n')

#
entry_url = Entry(label_frame_connect, font=(" None", 12))  # URL label
entry_url.place(relx=0.2, rely=0.090, anchor='n', height=30)
entry_username = Entry(label_frame_connect, font=(" None", 12))  # Password Label
entry_username.place(relx=0.2, rely=0.28, anchor='n', height=30)
entry_password = Entry(label_frame_connect, font=(" None", 12), show="*")  # Password Label
entry_password.place(relx=0.2, rely=0.5, anchor='n', height=30)

connect_bouton = Button(label_frame_connect, text="Connect", font=(12), command=connect)
connect_bouton.place(relx=0.2, rely=0.75, anchor='n')

# creation of query frame
label_frame_query = LabelFrame(window, text=" Query", font=(14))  # Query main Label
label_frame_query.place(relx=0.5, rely=0.40, relwidth=1, relheight=0.25, anchor='n')

label_query = Label(label_frame_query, text="Natural query", font=(12))  # Query main Label
label_query.place(relx=0.1, anchor='n')

entry_query = Text(label_frame_query, font=(" None", 12))  # Query main Label
entry_query.place(relx=0.25, rely=0.25, anchor='n', width=400, height=70)  # ,x=70,y=20

execute_bouton = Button(label_frame_query, text="Generate query", command=request, font=(12))
execute_bouton.place(relx=0.2, rely=0.7, anchor='n')

execute_bouton = Button(label_frame_query, text="Run", font=(14), command=execution)
execute_bouton.place(relx=0.80, rely=0.7, anchor='n')

## geration query
label_query_gen = Label(label_frame_query, text="Cypher query generated", font=(12))
label_query_gen.place(relx=0.60, anchor='n')
query_gen = StringVar()

query_gen = Text(label_frame_query, font=(" None", 12))
query_gen.place(relx=0.75, rely=0.25, anchor='n', width=500, height=65)  # x=70,y=30

cypher_response = StringVar()
cypher_response_lab = LabelFrame(window, text="Cypher_response", font=(14))
cypher_response_lab.place(relx=0.5, rely=0.65, relwidth=1, relheight=0.30, anchor='n')

# cypher_response=Label(cypher_response_lab,font=(14),bg="#CCFFCC")
# cypher_response.pack()

scroll_bar = Scrollbar(cypher_response_lab)
scroll_bar.pack(side=RIGHT, fill=Y)
mylist = Listbox(cypher_response_lab, yscrollcommand=scroll_bar.set)
mylist.place(relx=0.51, relwidth=1, anchor='n')
scroll_bar.config(command=mylist.yview)

scroll_v = Scrollbar(cypher_response_lab, orient=HORIZONTAL)
scroll_v.pack(side=BOTTOM, fill=X)
scroll_v.config(command=mylist.xview)
# log query
cypher_precision_lab = LabelFrame(cypher_response_lab, text="Accurency", font=(14))
cypher_precision_lab.place(relx=0.8, relwidth=0.5, relheight=1, anchor='n')
# cypher_response.place(relx=0.5,rely=0.80,relwidth=1,relheight=0.30,anchor='n')

text_accuracy = Text(cypher_precision_lab, foreground="red")  # Password Label
text_accuracy.pack(side=BOTTOM, fill=X)

# scroll_bar_accuracy = Scrollbar(cypher_precision_lab)
# scroll_bar_accuracy .pack(side=RIGHT,fill=Y)
# list_accuracy = Listbox(cypher_precision_lab,yscrollcommand=scroll_bar_accuracy.set)
# list_accuracy.place(relx=0.51,relwidth=1,anchor='n')
# scroll_bar_accuracy .config(command=list_accuracy.yview)
# scroll_v_accuracy = Scrollbar(cypher_precision_lab,orient= HORIZONTAL)
# scroll_v_accuracy.pack(side=BOTTOM,fill=X)
# scroll_v_accuracy.config(command = list_accuracy.xview)

# log query
cypher_log_lab = LabelFrame(cypher_response_lab, text="History", font=(14))
cypher_log_lab.place(relx=0.9, relwidth=0.4, relheight=1, anchor='n')

scroll_bar_log = Scrollbar(cypher_log_lab)
scroll_bar_log.pack(side=RIGHT, fill=Y)
list_log = Listbox(cypher_log_lab, yscrollcommand=scroll_bar_log.set)
list_log.place(relx=0.51, relwidth=1, anchor='n')
scroll_bar_log.config(command=list_log.yview)
scroll_v_log = Scrollbar(cypher_log_lab, orient=HORIZONTAL)
scroll_v_log.pack(side=BOTTOM, fill=X)
scroll_v_log.config(command=list_log.xview)
# cypher_response.place(relx=0.5,rely=0.80,relwidth=1,relheight=0.30,anchor='n')


# Query main Label
# query_gen.insert(0,request)
# v.set(reponse)
# msg = Entry(window, textvariable=v)
# msg.pack()

# Label(master, text='First Name').
# Label(master, text='Last Name').grid(row=1)
# e1 = Entry(master)
# e2 = Entry(master)
# e1.grid(row=0, column=1)
# e2.grid(row=1, column=1)

# lancement de la fenetre
window.mainloop()

#
