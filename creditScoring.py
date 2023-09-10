# https://app.datacamp.com/workspace/w/421ca202-c0b3-41cb-91e8-03ff99cc981b

# https://www.datacamp.com/tutorial/random-forests-classifier-python

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import tkinter as tk
import os
import xgboost as xgb
import seaborn as sns
import statsmodels.api as sm
import datetime
import pymssql

# from numpy.compat import basestring
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from datetime import datetime, timedelta
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from tabulate import tabulate

# from scipy.stats import norm

days_in_dates_first = True
save_pickle_name = "df.pkl"
save_csv_name = "training_data"
save_csv_single_correl = "single_correlations.csv"

# Default tree parameters
params_trees = {'number of trees': 100,
                'max depth of tree': 10,
                'min samples to split node': 10,
                'min samples to be in leaf': 5,
                'share training set': 0.7,
                'confusion matrix cutoff': 0.2,
                'show feature importance': True,
                'show ROC': True,
                'draw and save trees': False,
                'save the data': False,
                'save the model': True}

# Default lg parameters
params_lr = {'number of iterations': 100,
             'share training set': 0.7,
             'collinearity threshold': 0.65,
             'insignificance threshold': 0.025,
             'confusion matrix cutoff': 0.2,
             'auto feature elimination': False,
             'p-value cutoff': 0.1,
             'show ROC': True,
             'save the data': False,
             'save the model': True}

# Default kMeans parameters
params_kMeans = {'min groups': 2,
                 'max groups': 10,
                 'number random initiations': 10,
                 'show elbow chart': True,
                 'save the data': True,
                 'save the model': True}

rnd_state = 1
number_trees_pics = 5
trees_pics_depth = 2

# https://stackoverflow.com/questions/17098654/how-to-reversibly-store-and-load-a-pandas-dataframe-to-from-disk
save_data_pickle = False
load_model = False
get_data_from = "csv"  # "csv" or "pickle"

root = None
df_list = []  # an empty list of df objects
df_names = []  # an empty list of df object names
df = None
label_target_var = None
df_in_use = -1
operator_array = ["+", "-", "*", "/", "&", "="]
connector_array = ["=", "AND", "OR", "Cancel"]
lag_array = ["value", "min", "max", "sum", "mean", "std"]
overwrite_nan = -999999
unique_cutoff = 0.1
# if there are less than [x] unique values in a column, all will be shown
top_unique_value_threshold = 20
show_ROC = True
param_dict = []
auc_dict = {}
fpr_dict = {}
tpr_dict = {}


def main_menu():
    global root

    # Check if the root window already exists
    if root is not None:
        # Clear the existing buttons from the window
        for child in root.winfo_children():
            child.destroy()
    else:
        # Create a new root window if one doesn't exist
        root = tk.Tk()
        root.title("Main Menu")
        root.geometry("400x655")

    # Create a list of dictionaries with text and command for each button
    button_dicts = [{"text": "Load data", "command": load_data}]

    button_dicts.append(
        {"text": "Load data From Reporting Database ", "command": load_from_database})

    # Add data frame buttons
    button_dicts = core_menu(button_dicts)

    button_dicts.append({"text": "Quit", "command": root.destroy})
    button_dicts.append({"text": "Test", "command": test})

    # Determine the width of the widest label
    max_width = max(max([len(button["text"]) for button in button_dicts]), 20)

    # Create the buttons using a loop
    for button_dict in button_dicts:
        button = tk.Button(
            root, text=button_dict["text"], width=max_width, command=button_dict["command"])
        button.pack(side='top', anchor='w')

    root.mainloop()


def core_menu(buttons):
    """
    Adds buttons to the main menu for operations that require a dataframe to be loaded
    :param buttons: a list of dictionaries with text and command for each button
    :return: a list of dictionaries with text and command for each button
    """
    if df is not None:
        buttons.append({"text": "Save data", "command": save_data})
        buttons.append({"text": "Merge with...", "command": merge_data})
        buttons.append({"text": "Data summary", "command": info_stats}),
        buttons.append({"text": "Sort data", "command": sort_data})
        buttons.append({"text": "Set target variable",
                       "command": set_target_var})
        buttons.append({"text": "Time transformation",
                       "command": transform_times})
        buttons.append({"text": "Create dummies", "command": create_dummies})
        buttons.append({"text": "Rolling averages",
                       "command": previous_averages})
        buttons.append({"text": "Rolling values lags",
                       "command": rolling_window_lags})
        buttons.append({"text": "Delta Calculation",
                       "command": calc_delta_abs})
        buttons.append({"text": "Column operations", "command": column_calcs})
        buttons.append({"text": "Multi column operations",
                       "command": multi_column_calcs})
        buttons.append({"text": "Simple math transformations",
                       "command": simple_math_transformation})
        buttons.append({"text": "If...", "command": if_comparison3})
        buttons.append({"text": "Rename columns", "command": rename_columns})
        buttons.append({"text": "Delete columns", "command": remove_columns})
        buttons.append({"text": "Replace NaN", "command": replace_nan})
        buttons.append({"text": "Single Correlations",
                       "command": single_correlations})
        buttons.append({"text": "Correlation Matrix",
                       "command": correlation_matrix_chart})
        buttons.append({"text": "Train model", "command": choose_algorithm})
        buttons.append({"text": "Apply model", "command": apply_model})
    return buttons


def checkbox_menu(title_text, options):
    # create the top-level window for the selection dialog
    top = tk.Toplevel()
    top.title(title_text)
    top.geometry("400x600")

    # create a canvas with a scrollbar
    canvas = tk.Canvas(top, height=500)
    scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))
    canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))

    # create a dictionary to hold the IntVar for each checkbox
    vars_dict = {}
    for option in options:
        vars_dict[option] = tk.IntVar()

    # create the checkboxes
    for option in options:
        tk.Checkbutton(frame, text=option,
                       variable=vars_dict[option]).pack(anchor="w")

    # create the ok button to return the selected items
    def ok():
        selected_items = [option for option,
                          var in vars_dict.items() if var.get() == 1]
        top.destroy()
        return selected_items

    # create the cancel function to close the dialog
    def cancel():
        top.destroy()
        return []

    tk.Button(frame, text="OK", command=ok, width=10).pack(
        side="left", pady=10, padx=20, anchor="sw")
    tk.Button(frame, text="Cancel", command=cancel, width=10).pack(
        side="left", pady=10, padx=20, anchor="sw")

    # for more than one button
    # for button in buttons:
    #     tk.Button(frame, text=button, command=lambda text=button: ok(text), width=10).pack(
    #         side="left", pady=10, padx=20, anchor="sw")

    # wait for the selection dialog window to close
    top.wait_window()
    # return the selected items
    return ok()


def radio_menu(title_text):
    top = tk.Toplevel()
    top.title(title_text)
    top.geometry("800x600")

    # create a canvas with a scrollbar
    canvas = tk.Canvas(top, height=500)
    scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))
    scrollbar.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))

    options = df.columns.tolist()
    vars_dict = {}
    max_width = max(max([len(option) for option in options]), 20)

    for option in options:
        data_type = df[option].dtype
        unique_ratio = df[option].nunique() / len(df[option])
        if option == label_target_var:
            vars_dict[option] = tk.StringVar(value="ignore")
        elif option[:2].lower() == "id" or option[-2:].lower() == "id":
            vars_dict[option] = tk.StringVar(value="ignore")
        elif data_type == "int64" or data_type == "float64":
            vars_dict[option] = tk.StringVar(value="numerical")
        elif unique_ratio < unique_cutoff:
            vars_dict[option] = tk.StringVar(value="categorical")
        else:
            vars_dict[option] = tk.StringVar(value="ignore")

    for option in options:
        frame2 = tk.Frame(frame)
        frame2.pack(side="top", fill="x", padx=5, pady=5)
        label = tk.Label(frame2, text=option, width=max_width)
        label.pack(side="left")

        radio_frame = tk.Frame(frame2)
        radio_frame.pack(side="left", padx=5)

        for value in ["categorical", "numerical", "ignore"]:
            rb = tk.Radiobutton(radio_frame, text=value,
                                variable=vars_dict[option], value=value)
            rb.pack(side="left")

    def ok():
        selected_items = {option: var.get()
                          for option, var in vars_dict.items()}
        top.destroy()
        return selected_items

    def cancel():
        top.destroy()
        print("test")
        return {}

    tk.Button(frame, text="OK", command=ok, width=10).pack(
        side="bottom", pady=10, padx=20)
    tk.Button(frame, text="Cancel", command=cancel, width=10).pack(
        side="bottom", pady=10, padx=20)

    top.wait_window()

    return ok()


def button_menu(title_text, button_list, button_min_width=10, window_size="400x600", show_datetime=True):
    global df

    # Define function to be called when a button is clicked
    def button_clicked(sel_button):
        # Set the value of the selected column to the label_target_var
        button_selected.set(sel_button)
        # Close the window
        window.destroy()

    # Define function to be called when the Cancel button is clicked
    def cancel():
        # Close the window
        window.destroy()
        return

    # Create a tkinter window
    window = tk.Toplevel(root)
    window.title(title_text)
    window.geometry(window_size)

    # create a canvas with a scrollbar
    canvas = tk.Canvas(window, height=500)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))

    # Create a StringVar to hold the selected column name
    button_selected = tk.StringVar()

    # Add a label to display the selected column name
    label = tk.Label(frame, textvariable=button_selected)
    label.pack()

    # Find the maximum width of the button labels
    max_width = max(
        max([len(button)
            for button in button_list if show_datetime or df[button].dtype == "datetime64[ns]"]),
        button_min_width)

    for button in button_list:
        if show_datetime or df[button].dtype == "datetime64[ns]":
            new_button = tk.Button(
                frame, text=button, command=lambda i=button: button_clicked(i), width=max_width)
            new_button.pack(anchor="w")

    tk.Button(frame, text="Cancel", command=cancel, width=10).pack(
        side="bottom", pady=10, padx=20)

    # Wait for the window to be closed
    window.wait_window()

    # Return the selected column name
    return button_selected.get()


def math_op_menu(title_text):
    global df

    # Define function to be called when a button is clicked
    def button_clicked(operator):
        # Set the value of the selected operator
        chosen_operator.set(operator)
        # Close the window
        window.destroy()

    # Create a tkinter window
    window = tk.Toplevel(root)
    window.title(title_text)

    # Create a StringVar to hold the selected column name
    chosen_operator = tk.StringVar()

    # Add a label to display the selected column name
    label = tk.Label(window, textvariable=chosen_operator)
    label.pack()

    # Add a button for each column in the dataframe
    for column in operator_array:
        button = tk.Button(window, text=column, height=5, width=10,
                           command=lambda col=column: button_clicked(col))
        button.pack(side="left")

    # Wait for the window to be closed
    window.wait_window()

    # Return the selected operator
    return chosen_operator.get()


def math_comp_menu(title_text, var_summary):
    # create the top-level window for the selection dialog
    top = tk.Toplevel()
    top.title(title_text)
    top.geometry("400x600")

    # create a canvas with a scrollbar
    canvas = tk.Canvas(top, height=500)
    scrollbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)

    # pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event, canvas=canvas: canvas.configure(
        scrollregion=canvas.bbox("all")))
    canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
        int(-1 * (event.delta / 120)), "units"))

    stat_headers = ["Min: ", "Max:", "Mean: ", "Median: ", "Std: "]
    options = ["between", "not between", "=", "!=", "<", "<=", ">", ">="]

    for i in range(5):
        var = tk.StringVar()
        label = tk.Label(frame, textvariable=var)

        var.set(f"{stat_headers[i]} {var_summary[i]}")
        label.pack()

    combo_box = ttk.Combobox(frame, values=options)
    combo_box.pack()

    entry1 = ttk.Entry(frame)
    entry2 = ttk.Entry(frame)

    def on_combobox_select(event):
        selected_option = combo_box.get()
        entry1.pack()
        if selected_option in ["between", "not between"]:
            entry2.pack()
        else:
            entry2.pack_forget()

    def ok():
        # Retrieve the selected option and values
        selected_option = combo_box.get()
        value1 = entry1.get()
        value2 = entry2.get() if selected_option in [
            "between", "not between"] else None

        # Store the values as attributes of the window object
        top.selected_option = selected_option
        top.value1 = value1
        top.value2 = value2

        # Close the window
        top.destroy()

        # Return the selected option and values
        return selected_option, value1, value2

    combo_box.bind("<<ComboboxSelected>>", on_combobox_select)

    tk.Button(frame, text="OK", command=ok, width=10).pack(
        side="left", pady=10, padx=20, anchor="sw")

    # wait for the selection dialog window to close
    top.wait_window()

    # return the selected items
    return top.selected_option, top.value1, top.value2


def get_input(info_text, number_input=True, pre_sel_string=""):
    root = tk.Tk()
    root.withdraw()
    if number_input:
        user_input = simpledialog.askinteger("Input", f"Enter {info_text}:")
    else:
        user_input = simpledialog.askstring(
            "Input", f"Enter {info_text}:", initialvalue=pre_sel_string)
    root.destroy()
    return user_input


def sort_data():
    global df
    sort = False
    ranks = []
    i = -1
    df_col_names = df.columns.tolist()
    while not sort:
        i += 1
        ranks.append(button_menu(
            f"Choose the variable with rank {i+1} for sorting. Click 'Sort' to start sorting.", ["Sort"] + df_col_names))

        if ranks[i] == "":
            return
        if ranks[i] == "Sort":
            if i > 0:    # one cannot click "Sort" at the start without having specified one column, at least
                sort = True
            else:
                i -= 1
            ranks.remove("Sort")
        else:
            df_col_names.remove(ranks[i])

    df.sort_values(by=ranks, inplace=True)

    messagebox.showinfo("Data Sort", "Data sorted as specified.")


def get_tree_params():
    # Create a dictionary to hold the parameter values
    param_values = {}

    # Create a tkinter window for parameter input
    window = tk.Toplevel(root)
    window.title("Tree Model Parameters")

    # Create the parameter entry widgets and labels
    entries = {}
    for i, (key, default_val) in enumerate(params_trees.items()):
        frame = tk.Frame(window)
        frame.pack(side='top', fill='x', padx=5, pady=5)
        label = tk.Label(frame, text=key, width=20)
        label.pack(side='left')

        if key in ['show feature importance', 'show ROC', 'draw and save trees', 'save the data', 'save the model']:

            # Create a checkbox for boolean values
            var = tk.BooleanVar(value=params_trees[key])
            checkbox = tk.Checkbutton(frame, variable=var)
            checkbox.pack(side='left', padx=5)
            entries[key] = var
        else:
            # Create an entry for numeric values
            entry = tk.Entry(frame)
            entry.insert(0, default_val)
            entry.pack(side='left', padx=5)
            entries[key] = entry

    # Create a submit button to close the window and return the parameter values
    submit_button = tk.Button(window, text='OK', command=lambda: submit_params(
        window, param_values, entries))
    submit_button.pack(side='bottom', pady=10)

    # Wait for the window to be closed
    window.wait_window()

    return param_values


def get_lr_params():
    # Create a dictionary to hold the parameter values
    param_values = {}

    # Create a tkinter window for parameter input
    window = tk.Toplevel(root)
    window.title("Logistic Regression Parameters")

    # Create the parameter entry widgets and labels
    entries = {}
    for i, (key, default_val) in enumerate(params_lr.items()):
        frame = tk.Frame(window)
        frame.pack(side='top', fill='x', padx=5, pady=5)
        label = tk.Label(frame, text=key, width=20)
        label.pack(side='left')

        if key in ['auto feature elimination', 'show ROC', 'save the data', 'save the model']:
            # Create a checkbox for boolean values
            var = tk.BooleanVar(value=params_lr[key])
            checkbox = tk.Checkbutton(frame, variable=var)
            checkbox.pack(side='left', padx=5)
            entries[key] = var
        else:
            # Create an entry for numeric values
            entry = tk.Entry(frame)
            entry.insert(0, default_val)
            entry.pack(side='left', padx=5)
            entries[key] = entry

    # Create a submit button to close the window and return the parameter values
    submit_button = tk.Button(window, text='OK', command=lambda: submit_params(
        window, param_values, entries))
    submit_button.pack(side='bottom', pady=10)

    # Wait for the window to be closed
    window.wait_window()

    print(param_values)

    return param_values


def get_kMeans_params():
    # Create a dictionary to hold the parameter values
    param_values = {}

    # Create a tkinter window for parameter input
    window = tk.Toplevel(root)
    window.title("k-means Parameters")

    # Create the parameter entry widgets and labels
    entries = {}
    for i, (key, default_val) in enumerate(params_kMeans.items()):
        frame = tk.Frame(window)
        frame.pack(side='top', fill='x', padx=5, pady=5)
        label = tk.Label(frame, text=key, width=20)
        label.pack(side='left')

        if key in ['show elbow chart', 'save the data', 'save the model']:

            # Create a checkbox for boolean values
            var = tk.BooleanVar(value=params_kMeans[key])
            checkbox = tk.Checkbutton(frame, variable=var)
            checkbox.pack(side='left', padx=5)
            entries[key] = var
        else:
            # Create an entry for numeric values
            entry = tk.Entry(frame)
            entry.insert(0, default_val)
            entry.pack(side='left', padx=5)
            entries[key] = entry

    # Create a submit button to close the window and return the parameter values
    submit_button = tk.Button(window, text='OK', command=lambda: submit_params(
        window, param_values, entries))
    submit_button.pack(side='bottom', pady=10)

    # Wait for the window to be closed
    window.wait_window()

    return param_values


def submit_params(window, param_values, entries):
    # Update the parameter values with the values from the entry widgets
    for key, entry in entries.items():
        if isinstance(entry, tk.Entry):
            # Convert numeric values to floats
            param_values[key] = float(entry.get())
        else:
            # Convert Boolean values to True or False
            param_values[key] = bool(entry.get())

    # Close the window
    window.destroy()


def set_target_var():
    global df, label_target_var
    label_target_var = button_menu("Choose the target variable.", df.columns)
    if label_target_var:
        messagebox.showinfo(
            "Target Variable", f"The target variable was set to '{label_target_var}'.")
    else:
        messagebox.showinfo("Target Variable", "No target variable selected.")
        return


def merge_data():
    global df, df_names, df_in_use, df_list

    if len(df_names) < 2:
        return

    # Create a copy using slicing >> df_names_other = df_names would just create a
    df_names_other = df_names[:]
    # reference to the same list

    del df_names_other[df_in_use]

    df_name_to_add = button_menu(
        "Choose which dataset shall be added.", df_names_other)
    df_to_add = df_list[df_names.index(df_name_to_add)]

    if df_to_add is None or df_to_add.empty:
        return

    identifier = button_menu("Choose the identifier.", df.columns)

    # Inner join: only rows with matching values in both input DataFrames are included in the merged DataFrame.

    # Left join: all rows from the left input DataFrame and matching rows from the right input DataFrame are included
    # in the merged DataFrame. If there are no matching rows in the right input DataFrame, the columns from the right
    # input DataFrame are filled with NaN values.

    # Right join: all rows from the right input DataFrame and matching rows from the left input DataFrame are included
    # in the merged DataFrame. If there are no matching rows in the left input DataFrame, the columns from the left
    # input DataFrame are filled with NaN values.

    # Outer join: all ows from both input DataFrames are included in the merged DataFrame. If there are no matching
    # rows in one of the input DataFrames, the columns from that input DataFrame are filled with NaN values.

    join_type = "left"
    df_names.append(df_names[df_in_use] + "_" +
                    join_type + "_join_" + df_name_to_add)
    df_list[df_in_use] = df
    df_in_use += 1
    df = pd.merge(df, df_to_add, on=identifier, how=join_type)

    # To merge on multiple columns, pass a list of column names to the on parameter: on=["column1", "column2"].

    df_list.append(df)

    messagebox.showinfo(
        "Data Merge", "Both datasets have been joined into a new dataset.")


def create_dummies():
    global df
    cols_to_dummy = checkbox_menu(
        "Choose which variables shall be binarized.", df.columns.tolist())
    min_dummy_number = get_input(
        "the minimum count of observations to qualify for binarization")
    delete_initial_column = button_menu("What shall happen with the initial column?",
                                        ["Keep initial column", "Remove initial column"])
    if min_dummy_number:
        for col in cols_to_dummy:
            dummy_variables(df, col, min_dummy_number)
            if delete_initial_column == "Remove initial column":
                df = df.drop(col, axis=1)

    if cols_to_dummy and min_dummy_number:
        messagebox.showinfo("Dummy Variables",
                            "Selected columns have been binarized.")
    else:
        messagebox.showinfo("Dummy Variables",
                            "No columns selected or no number provided.")


def simple_math_transformation():
    global df
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    cols_to_math = checkbox_menu(
        "Choose which variables shall be mathematically transformed.", numeric_cols)

    sel_math_op = button_menu("Select one option to be performed.",
                              ["+, -, * , / constant", "SQRT(x), 1/x, LN(x), x^2", "Rounding"])

    if sel_math_op == "Rounding":
        digits = get_input("How many digits?")
        for col in cols_to_math:
            df[col] = df[col].apply(lambda x: round(x, digits))
    elif sel_math_op == "SQRT(x), 1/x, LN(x), x^2":
        for col in cols_to_math:
            simple_math_trans_variables(df, col)
    else:
        operator = button_menu("Choose an operation.", operator_array[:4])
        const = get_input("Enter the number.")
        for col in cols_to_math:
            if operator == "+":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] + const)
            elif operator == "-":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] - const)
            elif operator == "*":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] * const)
            elif operator == "/":
                df.insert(loc=df.columns.get_loc(col) + 1, column='' + col + ' ' + operator + ' ' + str(const),
                          value=df[col] / const)

    messagebox.showinfo("Math transformation",
                        "The math transformations have been made.")


def transform_times():
    global df
    cols_to_time_trans = checkbox_menu(
        "Choose which variables shall be made time variables.", df.columns.tolist())
    time_trans_kinds = button_menu("test", ["All", "Only days since 1900"])
    for col in cols_to_time_trans:
        time_transformation(df, col, time_trans_kinds)

    if cols_to_time_trans:
        messagebox.showinfo("Time Transformation",
                            "Selected columns have been made time variables.")
    else:
        messagebox.showinfo("Time Transformation", "No columns selected.")


def column_calcs():
    operator = "x"
    first_column = None
    new_col_label = ""
    while not (operator == "="):
        if operator == "x":
            first_column_df = df[[button_menu(
                "Perform a standard maths operation on two columns. Choose the first column.",
                df.select_dtypes(include=['number']).columns)]]  # double [[]] to return df object incl. label name
            new_col_label = first_column_df.columns[0]
            first_column = first_column_df[first_column_df.columns[0]]
        operator = button_menu("Perform a standard maths operation on two columns. Choose the operator.",
                               operator_array)
        if not (operator == "="):
            second_column_df = df[
                [button_menu("Perform a standard maths operation on two columns. Choose the second column.",
                             df.select_dtypes(include=['number']).columns)]]
            new_col_label = "(" + new_col_label + "_" + \
                operator + "_" + second_column_df.columns[0] + ")"
            second_column = second_column_df[second_column_df.columns[0]]
            first_column = column_operation(
                first_column, second_column, operator)
    df.insert(len(df.columns), new_col_label, first_column)
    #              array.pop(new_column_name))
    # if first_column and operator and second_column and operator == "=":
    messagebox.showinfo("Column Operations",
                        f"A new column {new_col_label} was created.")
    # else:
    #     messagebox.showinfo("Column Operations", "Something went wrong.")


def multi_column_calcs():
    second_columns = None
    first_columns = checkbox_menu("Perform a standard maths operation on two columns. Choose the first columns.",
                                  df.columns.tolist())
    operator = button_menu(
        "Perform a standard maths operation on two columns. Choose the operator.", operator_array)
    if not (operator == "="):
        second_columns = checkbox_menu("Perform a standard maths operation on two columns. Choose the second columns.",
                                       df.columns.tolist())
        multi_column_operation(df, first_columns, second_columns, operator)
    if first_columns and not (operator == "=") and second_columns:
        messagebox.showinfo("Multi column Operations",
                            "New columns have been created.")
    else:
        messagebox.showinfo("Multi column operations", "Something went wrong.")


def rename_columns():
    global df
    col_to_rename = button_menu(
        "Choose which variable shall be renamed.", df.columns)
    if not col_to_rename:
        return

    new_name = get_input("the new column name", False)
    df = df.rename(columns={col_to_rename: new_name})
    if col_to_rename:
        messagebox.showinfo(
            "Renamed Column", f"Old name: {col_to_rename}; new name: {new_name}")
    else:
        messagebox.showinfo("Renamed Column", "No column was renamed.")


def remove_columns():
    global df
    cols_to_del = checkbox_menu(
        "Choose which variables shall be removed.", df.columns.tolist())
    for col in cols_to_del:
        df = df.drop(col, axis=1)

    if cols_to_del:
        messagebox.showinfo("Removed Columns",
                            "Selected columns have been deleted.")
    else:
        messagebox.showinfo("Removed Columns", "No columns deleted.")


def previous_averages():
    global df
    # check if a time column exists
    if not any(df.dtypes == 'datetime64[ns]'):
        messagebox.showinfo("Error",
                            "This cannot be executed. First specify a time column using 'Time transformation'.")
    else:
        target_cols = checkbox_menu("Choose on which variables rolling counts, sums, averages and SDs be performed.",
                                    df.columns.tolist())
        time_ref_col = button_menu(
            "Choose to which time variable to refer.", df.columns, show_datetime=False)
        condition_col = button_menu(
            "Choose which variable contains the condition (typically an id).", df.columns)
        days_back = get_input("the days to look backwards")
        if days_back:
            for col in target_cols:
                rolling_window(df, col, time_ref_col, condition_col, days_back)

        if target_cols and days_back:
            messagebox.showinfo(
                "Rolling Window", "Rolling operations have been performed on selected columns.")
        else:
            messagebox.showinfo(
                "Rolling Window", "No columns selected or no number provided.")


def calc_delta_abs():
    global df
    target_cols = checkbox_menu(
        "Choose on which variables the differences shall be calculated.", df.columns.tolist())
    condition_col = button_menu(
        "Choose which variable contains the condition (typically an id).", df.columns)
    rows_up = get_input("the rows to look upwards (typically 1)")
    if rows_up:
        for col in target_cols:
            delta(df, col, condition_col, rows_up)

    if target_cols and rows_up:
        messagebox.showinfo(
            "Delta Calculation", "Differences on the selected columns have been calculated.")
    else:
        messagebox.showinfo("Delta Calculation",
                            "No columns selected or no number provided.")


def replace_nan():
    global df
    repl_value = get_input("a number to replace 'NaN' and 'Div/0' with")
    nan_replacement(df, repl_value)
    messagebox.showinfo(
        "NaN Replacement", f"All 'NaN' and 'Div/0' have been replaced with {repl_value}.")


def single_correlations():
    global df

    if not label_target_var:
        set_target_var()
        if not label_target_var:
            return

    single_correl_selection = radio_menu(
        "Choose the datatype to get the tables.")
    if not single_correl_selection:
        print(single_correl_selection)
        return

    print(single_correl_selection)
    min_count = get_input(
        "the minimum number of categorical observations to get shown")
    num_buckets = get_input(
        "into how many buckets continuous variables shall be split")
    part1 = single_correl_num(
        df, single_correl_selection, num_buckets, label_target_var)
    single_correl_cat(df, single_correl_selection, min_count,
                      num_buckets, label_target_var, part1)

    messagebox.showinfo("Single Correlations",
                        "A csv file showing the single target correlations was created.")


def correlation_matrix_chart():
    global df
    # correl_matrix = df.corr(numeric_only=True)
    correl_matrix = df.corr(numeric_only=False)

    # Create a custom color map
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.set(rc={'figure.figsize': (20, 20)})
    sns.heatmap(correl_matrix, annot=True, cmap=cmap,
                center=0, fmt=".2f", annot_kws={"size": 10})

    # Rotate x-axis labels and align them to the right
    plt.xticks(rotation=45, ha="right")
    # Adjust layout and add more space for labels
    plt.tight_layout(rect=[0.15, 0.15, 1, 0.85])
    plt.show()


def choose_algorithm():
    global show_ROC

    which_model = button_menu("Which algorithm shall be used?",
                              ["All Classification", "Random Forest", "Extra Trees", "AdaBoost", "Gradient Boost",
                               "XGBoost", "Logistic Regression", "k-means"])
    if which_model == "All Classification":
        tree_algorithms = [RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier,
                           xgb.XGBClassifier]
        get_params = True
        all_classification = True
        for clf in tree_algorithms:
            train_tree_classifier(clf, get_params, all_classification)
            get_params = False
        train_logistic_regression(all_classification)
    elif which_model == "Random Forest":
        train_tree_classifier(RandomForestClassifier)
    elif which_model == "Extra Trees":
        train_tree_classifier(ExtraTreesClassifier)
    elif which_model == "AdaBoost":
        train_tree_classifier(AdaBoostClassifier)
    elif which_model == "Gradient Boost":
        train_tree_classifier(GradientBoostingClassifier)
    elif which_model == "XGBoost":
        train_tree_classifier(xgb.XGBClassifier)
    elif which_model == "Logistic Regression":
        train_logistic_regression()
    elif which_model == "k-means":
        train_kMeans()
    else:
        return

    # Convert the dictionary items to a list of lists (each key-value pair is a row)
    auc_table = [[key, value] for key, value in auc_dict.items()]
    fpr_table = [[key, value.tolist()] for key, value in fpr_dict.items()]
    tpr_table = [[key, value.tolist()] for key, value in tpr_dict.items()]

    # Display the table with tabulate
    print(tabulate(auc_table, headers=["Classifier", "AUC"], tablefmt="grid"))

    if show_ROC:
        for i in range(len(auc_table)):
            fpr_array = np.array(fpr_table[i][1])
            tpr_array = np.array(tpr_table[i][1])
            auc_value = auc_table[i][1]
            plt.plot(fpr_array, tpr_array, label="Classifier: {}, AUC: {}".format(
                auc_table[i][0], auc_value))

        plt.legend(loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.show()


def time_transformation(array, label_name, time_trans_kinds):
    """
    transforms columns with dates into new columns showing the year, the month, the day and the weekday
    :param array: the data frame object (df)
    :param label_name:
    :param time_trans_kinds:
    :return:
    """
    global df

    # careful - always check what is month and what is day and swap if necessary

    col_position = array.columns.get_loc(label_name)
    array[label_name] = pd.to_datetime(
        array[label_name], dayfirst=days_in_dates_first)
    array.insert(col_position, label_name + "_daysSince1900", array[label_name].astype(
        np.int64) // 10 ** 9 / 24 / 60 / 60 + 25569)  # converts the values to seconds since the Unix epoch (January 1, 1970) and then back into an Excel-like number_format

    if time_trans_kinds == "All":
        array.insert(col_position, label_name + "_hour",
                     pd.to_datetime(array[label_name]).dt.hour, True)
        array.insert(col_position, label_name + "_weekday", pd.to_datetime(array[label_name]).dt.weekday + 1,
                     True)  # + 1 because in Python Monday is 0 and Sunday is 6
        array.insert(col_position, label_name + "_day",
                     pd.to_datetime(array[label_name]).dt.day, True)
        array.insert(col_position, label_name + "_month",
                     pd.to_datetime(array[label_name]).dt.month, True)
        array.insert(col_position, label_name + "_year",
                     pd.to_datetime(array[label_name]).dt.year, True)
    else:
        array = array.drop(label_name, axis=1)

    df = array


def rolling_window(array, target_column, time_ref_column, condition_column, days_backwards):
    """
    creates previous time counts, sums, averages and SDs on a specified column and a specified time to look backwards
    :param array: the data frame object (df)
    :param target_column: the target_column on which the operations shall be performed
    :param time_ref_column: the column which contains the time stamp of the transaction or view
    :param condition_column: the column which is the =SUMIF(); =AVERAGEIF(), etc. condition, e.g customer_id
    :param days_backwards: the number of days looking backwards to sum, average, etc.
    :return:
    """
    global df
    rolling_column = \
        array.set_index(time_ref_column).groupby(condition_column).rolling(window=str(days_backwards) + 'D',
                                                                           closed="left")[
            target_column].agg(['count', 'sum', 'mean', 'max', 'min', 'std']).reset_index()

    array = pd.merge(array, rolling_column, on=[
                     time_ref_column, condition_column])

    df = array

    df.rename(columns={"count": target_column + "_count_prev" + str(days_backwards) + "d",
                       "sum": target_column + "_sum_prev" + str(days_backwards) + "d",
                       "mean": target_column + "_mean_prev" + str(days_backwards) + "d",
                       "max": target_column + "_max_prev" + str(days_backwards) + "d",
                       "min": target_column + "_min_prev" + str(days_backwards) + "d",
                       "std": target_column + "_std_prev" + str(days_backwards) + "d"}, inplace=True)

    # FutureWarning: Dropping of nuisance columns in rolling operations is deprecated; in a future version this will
    # raise TypeError. Select only valid columns before calling the operation. Dropped columns were Index(['dob'],
    # dtype='object')

    # careful, currently the merging seems to create some duplicates if there is more than one transaction with same condition and same time stamp

    # array[avg_column_name] = array[avg_column_name].fillna(-1)  # replace #N/A with -1


def rolling_window_lags():
    """
    returns values, min, max, sums, averages and SDs on a specified column and a specified backwards period (lag)
    naming convention: l[lags upwards]_[function]_[column reference]-[divider]_[div column reference]
    the first column reference is the column the operation gets applied on, the second the column divided through
    the column divided through refers to the previous values, i.e. from one lag (row) above
    """
    global df

    num_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    # time_cols = df.select_dtypes(include=['datetime64[ns]', 'timedelta64[ns]', 'period[D]', 'int', 'float']).columns.tolist()

    target_cols = checkbox_menu(
        "Choose the columns to create lags on.", num_cols)
    if not target_cols:
        return
    operators = checkbox_menu("What shall be performed?", lag_array)
    if not operators:
        return
    num_lags = checkbox_menu("How many lags upwards?",
                             [i for i in range(1, 13)])
    if not num_lags:
        return
    abs_rel = checkbox_menu(
        "Return absolute or relative to the actual value?", ["abs", "rel"])
    if not abs_rel:
        return
    condition_col = button_menu(
        "What is the condition (typically an ID)?", df.columns.tolist())
    if not condition_col:
        return
    time_ref_col = button_menu(
        "Please specify the time column.", df.columns.tolist())
    if not time_ref_col:
        return

    if 'rel' in abs_rel:
        division_cols = checkbox_menu("Choose the columns to divide through for the relative lag columns.",
                                      num_cols + ["actual column"])
        # prev_actual = button_menu("Use the actual or the previous lag for division?", ["actual", "previous"])
        lags_up_div = 1
        # if prev_actual == "actual":
        #     lags_up_div = 0
        # else:
        #     lags_up_div = 1
        #
        # print(lags_up_div) # works well for previous but not for actual, as there it return the first and not the last value of the rolling window

    delete_originals = button_menu(
        "Shall the original columns be deleted?", ["yes", "no"])

    sort_data(df, condition_col, time_ref_col, None)

    # Iterate through each target column >> probably faster to change the for loops for col and lag
    for col in target_cols:
        results_df = None
        for lags in num_lags:

            # Group by the condition column and apply rolling operation
            rolling_df = df.groupby(condition_col)[col].rolling(
                window=lags, min_periods=1, closed='left')

            if 'rel' in abs_rel:
                rolling_div_dfs = {}  # Dictionary to store rolling_div_df for each division column

                if "actual column" in division_cols:
                    division_cols[division_cols.index("actual column")] = col

                for div in division_cols:
                    rolling_div_df = df.groupby(condition_col)[div].rolling(
                        window=lags, min_periods=1, closed='left')
                    rolling_div_dfs[div] = rolling_div_df

            for operator in operators:

                for a_r in abs_rel:

                    # Calculate the lagged metrics using the specified operators
                    if operator == "value":
                        lagged_metrics = rolling_df.apply(
                            lambda x: x.iloc[-lags] if len(x) >= lags else np.nan)
                    else:
                        lagged_metrics = rolling_df.agg(operator)

                    new_col_name = f"l{lags}_{operator}_{col}"

                    if a_r == "rel":
                        for div in division_cols:
                            rolling_div_df = rolling_div_dfs[div]
                            div_lagged_metrics = rolling_div_df.apply(
                                lambda x: x.iloc[-lags_up_div] if len(x) >= lags_up_div else np.nan)

                            lagged_metrics_new = lagged_metrics / div_lagged_metrics

                            new_col_name = f"l{lags}_{operator}_{col}_/_{div}"
                            lagged_metrics_new = lagged_metrics_new.rename(
                                new_col_name)
                            # Concatenate the lagged_metrics DataFrame with the results DataFrame
                            results_df = pd.concat(
                                [results_df, lagged_metrics_new], axis=1)

                    else:
                        lagged_metrics = lagged_metrics.rename(new_col_name)
                        # Concatenate the lagged_metrics DataFrame with the results DataFrame
                        results_df = pd.concat(
                            [results_df, lagged_metrics], axis=1)

        if 'rel' in abs_rel and col in division_cols:
            division_cols[division_cols.index(col)] = "actual column"

        # Get the index of the col column
        col_index = df.columns.get_loc(col)

        results_df = results_df.reset_index(drop=True)
        df = df.reset_index(drop=True)

        # Split the DataFrame columns and insert the lagged_metrics Series
        df = pd.concat([df.iloc[:, :col_index + 1], results_df,
                       df.iloc[:, col_index + 1:]], axis=1)

        if delete_originals == "yes":
            df.drop(col, axis=1, inplace=True)

    # Mark all but the first occurrence of each column as duplicates
    duplicates_mask = df.columns.duplicated(keep='first')

    # Select columns that are not duplicates
    df = df.loc[:, ~duplicates_mask]

    print(df)


def column_operation(col1, col2, operator):
    """
    Mathematically connects two columns by a chosen operator - "+", "-", "*" or "/"
    :param array: the data frame object (df)
    :param label1: the first column
    :param label2: the second column
    :param operator: "+", "-", "*" or "/"
    :return:
    """
    # global df
    calc_column = None
    # new_column_name = "test"
    # calc_column = pd.Series(dtype=float)  # a new column to store the new data
    # array.insert(array.columns.get_loc(label2) + 1, new_column_name,
    #              array.pop(new_column_name))  # move the new column from the end just after the avg_target column
    if operator == "+":
        calc_column = col1 + col2
    elif operator == "-":
        calc_column = col1 - col2
    elif operator == "*":
        calc_column = col1 * col2
    elif operator == "/":
        calc_column = col1 / col2
    elif operator == "&":
        calc_column = col1.astype(str).str.cat(col2.astype(str), sep="_")
    else:
        return
    return calc_column


def multi_column_operation(array, labels1, labels2, operator):
    """
    Mathematically connects two columns by a chosen operator - "+", "-", "*" or "/". Can be performed on multiple columns simultaneously.
    :param array: the data frame object (df)
    :param labels1: all first columns
    :param labels2: all second columns
    :param operator: "+", "-", "*" or "/"
    :return:
    """
    global df

    for i1 in labels1:
        for i2 in labels2:
            new_column_name = i1 + "_" + operator + "_" + i2
            # a new column to store the new data
            array[new_column_name] = pd.Series(dtype=float)
            array.insert(array.columns.get_loc(i2) + 1, new_column_name,
                         array.pop(
                             new_column_name))  # move the new column from the end just after the avg_target column
            if operator == "+":
                array[new_column_name] = array[i1] + array[i2]
            elif operator == "-":
                array[new_column_name] = array[i1] - array[i2]
            elif operator == "*":
                array[new_column_name] = array[i1] * array[i2]
            elif operator == "/":
                array[new_column_name] = array[i1] / array[i2]
            elif operator == "&":
                array[new_column_name] = array[i1].astype(
                    str).str.cat(array[i2].astype(str), sep="_")
    df = array


def info_stats():
    global df
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)
    print("Data")
    print()
    print(df.head(10000))
    print()
    print('The shape of our data is (rows x columns):', df.shape)
    print()
    print("Percentiles, etc.")
    print(df.describe())
    print()
    print("Column names, non-null counts and data types per column")
    print(df.info())
    print()
    print("Number of unique values per column")
    print(df.nunique())
    print()
    main_menu()


def remove_unwanted_datatypes(array):
    """
    removes the target variable and unwanted datatypes which cannot be handled by random forest
    :param array: the data frame object (df)
    :return:
    """
    print('label_target_var = ', label_target_var)
    if label_target_var is not None:
        # removes the target variable from the feature list
        array = array.drop(label_target_var, axis=1)
    # print('array=', array)
    # print(type(array))
    array = array.select_dtypes(exclude=['object', 'datetime64[ns]'])
    # print('array after drop object datetime64[ns]=', array)
    # Identify and remove constant columns
    # Select only non-constant columns
    array = array[[col for col in array.columns if array[col].nunique() > 1]]
    # print('Select only non-constant columns from array', array)

    return array


def dummy_variables(array, column_to_dummy, min_number_values):
    """
    creates dummy variables [0, 1] of specified columns and inserts them at the end; removes the original column
    :param array:
    :param column_to_dummy:
    :param min_number_values: the minimum number of expressions to qualify for a single dummy column
    :return:
    """

    global df

    cats = array[column_to_dummy].value_counts(
    )[lambda x: x >= min_number_values].index

    df = pd.concat(
        [array, pd.get_dummies(pd.Categorical(
            array[column_to_dummy], categories=cats), prefix=column_to_dummy)],
        axis=1)


def simple_math_trans_variables(array, col_to_calc):
    """
    Calculates SQRT(col_to_calc), 1/col_to_calc, LN(col_to_calc) and col_to_calc^2 on specified columns and inserts them at the end; does not remove the original column
    :param array:
    :param col_to_calc:
    :return:
    """
    global df
    # think of adding 1 or more to avoid Div/0, LN(0), LN(neg), SQRT(neg)

    sqrt_col = np.sqrt(array[col_to_calc])
    inv_col = 1 / array[col_to_calc]
    ln_col = np.log(array[col_to_calc])
    square_col = array[col_to_calc] ** 2

    # insert the new columns after col_to_calc
    array.insert(loc=array.columns.get_loc(col_to_calc) + 1,
                 column='SQRT(' + col_to_calc + ')', value=sqrt_col)
    array.insert(loc=array.columns.get_loc(col_to_calc) + 2,
                 column='1/' + col_to_calc, value=inv_col)
    array.insert(loc=array.columns.get_loc(col_to_calc) + 3,
                 column='LN(' + col_to_calc + ')', value=ln_col)
    array.insert(loc=array.columns.get_loc(col_to_calc) + 4,
                 column=col_to_calc + '^2', value=square_col)

    df = array


def if_comparison():
    global df
    sel_col_label = button_menu("Select a column.", df.columns.tolist())
    sel_col = df[sel_col_label]

    num_unique = sel_col.nunique()

    show_threshold = top_unique_value_threshold

    if sel_col.dtype == object or num_unique <= 5:

        if num_unique > show_threshold:
            show_threshold = get_input(
                "the number of top value counts to be shown")

        # Get the unique values and their counts in the column
        value_counts = sel_col.value_counts()

        # Get the top values based on count of appearance
        unique_values = value_counts.index.tolist()[:show_threshold]

        sel_expressions = checkbox_menu(
            "Select the expressions.", unique_values)
        logic = button_menu("Choose the logic.", ["= / OR", "NOT"])
        new_col_name = "multiple"
        if len(sel_expressions) < 5:
            new_col_name = '_'.join(str(expr) for expr in sel_expressions)

        if logic == "OR":
            df[sel_col_label + "_" + logic + "_" +
                new_col_name] = df[sel_col_label].isin(sel_expressions).astype(int)
        else:
            df[sel_col_label + "_" + logic + "_" + new_col_name] = (~df[sel_col_label].isin(sel_expressions)).astype(
                int)

        new_col_name = sel_col_label + "_" + logic + "_" + new_col_name

    else:
        # Get summary statistics for the numeric column
        summary_stats = [np.min(sel_col), np.max(sel_col), np.mean(sel_col), np.median(sel_col),
                         np.std(sel_col)]
        comp_choice, value1, value2 = math_comp_menu(
            "Choose the logic.", summary_stats)
        value1 = float(value1)
        if value2 is not None:
            value2 = float(value2)

            if value2 < value1:
                value_temp = value2
                value2 = value1
                value1 = value_temp

        if comp_choice == "between":
            df[f"{sel_col_label}_{comp_choice}_{value1}_{value2}"] = (
                (sel_col >= float(value1)) & (sel_col <= float(value2))).astype(int)
        elif comp_choice == "not between":
            df[f"{sel_col_label}_{comp_choice}_{value1}_{value2}"] = (
                ~(sel_col.between(float(value1), float(value2)))).astype(int)
        elif comp_choice == "=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col == float(value1)).astype(int)
        elif comp_choice == "!=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col != float(value1)).astype(int)
        elif comp_choice == "<":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col < float(value1)).astype(int)
        elif comp_choice == "<=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col <= float(value1)).astype(int)
        elif comp_choice == ">":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col > float(value1)).astype(int)
        elif comp_choice == ">=":
            df[f"{sel_col_label}_{comp_choice}_{value1}"] = (
                sel_col >= float(value1)).astype(int)

        new_col_name = sel_col_label + "_" + comp_choice + "_" + str(value1)
        if value2 is not None:
            new_col_name = new_col_name + "_" + str(value2)

    messagebox.showinfo(
        "New Column", f"Column {new_col_name} has been created.")

    return


def delta(array, ref_column, condition_column, look_upwards):
    """
    calculates the difference (delta) to the previous value of the same column
    :param array: the data frame object (df)
    :param ref_column: the column the delta shall be calculated on
    :param condition_column: the column indicating the condition, e.g. customer_id
    :param look_upwards: the number of rows to look up
    """
    global df

    # calculate absolute difference
    array[ref_column + "_delta_abs"] = array.groupby(
        condition_column)[ref_column].diff(periods=look_upwards)

    # calculate relative difference

    array[ref_column + "_delta_rel"] = (
        array.groupby(condition_column, group_keys=False)[ref_column]
        .apply(lambda x: x.pct_change(periods=look_upwards))
    )

    df = array


def load_from_database():
    global df, df_in_use, df_names

    conn = pymssql.connect(server='T24DB-REPORTS'                           # ,user='zeppelin'
                           # ,password='zeppelin'
                           , database='His_T24_Reports')

    cursor = conn.cursor(as_dict=True)

    cursor.callproc('dbo.FatenDataSet')
    rows = []
    for row in cursor:
        rows.append(list(row.values())[0:])

    # Save all Headers for SQL
    columns = [column[0] for column in cursor.description]

    # Creating an empty dict
    myDict = dict()

    # Creating a list
    valList = columns[1:]

    k = 0
    # Iterating the elements in list
    for val in valList:
        for ele in range(0, len(rows)-1, 1):
            myDict.setdefault(val, []).append(rows[ele][k])
        k = k+1
    # print(np.asarray(myDict))

    # Close cursor and connection
    cursor.close()
    conn.close()

    df = pd.DataFrame(myDict)
    # print(df.columns)
    # print('x:-', df)
    messagebox.showinfo("Information", "Data successfully loaded.")
    # return rows
    main_menu()


def load_data():
    global df, df_in_use, df_names

    if df is not None and not df.empty:
        load_source = button_menu("Data source?", ["Data Frame", "File"])
    else:
        load_source = "File"

    if not load_source:
        return

    if df is None or df.empty or load_source == "File":
        root = tk.Tk()
        root.withdraw()

        # Set the starting directory to the current working directory
        start_dir = os.getcwd()

        # Show the file selection dialog box and get the selected file
        file_path = filedialog.askopenfilename(initialdir=start_dir, title="Select a file",
                                               filetypes=(("CSV files", "*.csv"),
                                                          ("Excel files", ("*.xls",
                                                           "*.xlsx", "*.xlsb")),
                                                          ("All files", "*.*")),
                                               initialfile="*.csv;*.xls;*.xlsx;*.xlsb")

        root.destroy()
        if not file_path:
            messagebox.showinfo("Information", "No file selected.")
        else:
            # Load the selected file as a pandas dataframe
            if df_in_use > -1:
                df_list[df_in_use] = df
            df_in_use += 1

            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx', '.xlsb']:
                df = pd.read_excel(file_path)
            else:
                messagebox.showinfo("Information", "Unsupported file format.")
                return

            print("Selected file:", file_path)
            print("Dataframe shape:", df.shape)

            df_names.append(get_input("a name for the dataset", False,
                            os.path.splitext(os.path.basename(file_path))[0]))
            df_list.append(df)
            messagebox.showinfo("Information", "Data successfully loaded.")

    else:
        load_df = button_menu("Choose the data you want to load?", df_names)
        if not load_df:
            return
        else:
            df = df_list[df_names.index(load_df)]
            messagebox.showinfo("Information", "Data successfully loaded.")

    main_menu()


def save_data():
    global df, df_in_use, df_names

    saving_type = button_menu("How to save?", ["Data Frame", "CSV File"])

    if not saving_type:
        return

    elif saving_type == "CSV File":
        root = tk.Tk()
        root.withdraw()

        # Set the starting directory to the current working directory
        start_dir = os.getcwd()

        # Show the file save dialog box and get the selected file name and location
        file_path = filedialog.asksaveasfilename(initialdir=start_dir, title="Save as...",
                                                 filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        root.destroy()
        if not file_path:
            messagebox.showinfo("Information", "Document not saved.")
        else:
            # Check if the file path has the .csv extension
            if not file_path.lower().endswith('.csv'):
                # Append .csv extension to the file path
                file_path += '.csv'
            # Save the dataframe as a CSV file at the selected location
            df.to_csv(file_path, index=False)
            messagebox.showinfo(
                "Information", f"Data successfully saved in '{file_path}'.")
            print("Saved file:", file_path)

    else:
        if df_names:
            overwrite = button_menu(
                "Where to save?", ["Actual_"+df_names[df_in_use], "New"])
        else:
            overwrite = "New"
        if not overwrite:
            return
        elif overwrite == "New":
            df_names.append(get_input("a name for the dataset", False))
            df_list.append(df)
            df_in_use += 1
        else:
            df_list[df_in_use] = df

        messagebox.showinfo("Information", "Data saved internally.")


def nan_replacement(array, replacement_value):
    global df
    array.replace(to_replace=np.nan, value=replacement_value, inplace=True)
    array.replace(to_replace=np.inf, value=replacement_value, inplace=True)
    array.replace(to_replace=-np.inf, value=replacement_value, inplace=True)
    df = array


def single_correl_num(df, radio_array, n, target_col):
    """Find the n boundaries for each column in the given pandas dataframe.

    Args:
        df: pandas DataFrame object with numerical and non-numerical columns
        n: number of buckets to split the data into (default 10)
        target_var: name of the column to count '1' values in (default None)
        output_file: path to CSV file to write the bucket information to (default None)

    Returns:
        A dictionary with column names as keys and a list of n split values as values.
    """
    result = {}
    bucket_info = []
    for col in df.columns:
        if df[col].dtype.kind in 'biufc' and radio_array.get(col, "") == "numerical" and col != target_col and df[
                col].nunique() > n:

            # overwrite blanks with a large negative value or so
            df[col] = df[col].fillna(overwrite_nan)

            boundaries = np.percentile(df[col], np.linspace(0, 100, n + 1))
            result[col] = list(boundaries)
            counts, _ = np.histogram(df[col], bins=boundaries)
            bucket_info.append([col, "Count", "Sum of 1s", "Average"])
            total_bucket_count = 0
            total_target_count = 0
            for i, bucket in enumerate(range(1, n + 1)):
                lower_bucket = result[col][i]
                upper_bucket = result[col][i + 1]
                bucket_count = counts[i]
                total_bucket_count += bucket_count

                if i < n - 1:
                    bucket_values = df[col][(df[col] >= lower_bucket) & (
                        df[col] < upper_bucket)]
                    upper_bucket_label = "<"
                else:
                    bucket_values = df[col][
                        (df[col] >= lower_bucket) & (df[col] <= upper_bucket)]  # the last bucket shall include <=
                    upper_bucket_label = "<="

                # can't handle dates

                target_count = len(
                    bucket_values[bucket_values.index.isin(df[df[target_col] == 1].index)])
                total_target_count += target_count
                if upper_bucket > lower_bucket:  # prevents incl. empty rows
                    if col[
                       -13:] != "_daysSince1900":  # shall ensure that dates are presented like dates in the else statement
                        bucket_info.append(
                            [f"{lower_bucket:.2f}_-_{upper_bucket_label}{upper_bucket:.2f}", bucket_count, target_count,
                             target_count / bucket_count])
                    else:
                        dt = datetime(1899, 12, 30) + \
                            timedelta(days=int(lower_bucket))
                        date_str_lower = dt.strftime('%d.%m.%Y')
                        dt = datetime(1899, 12, 30) + \
                            timedelta(days=int(upper_bucket))
                        date_str_upper = dt.strftime('%d.%m.%Y')
                        bucket_info.append(
                            [f"{date_str_lower}_-_{upper_bucket_label}{date_str_upper}", bucket_count, target_count,
                             target_count / bucket_count])

            bucket_info.append(
                ["Total", total_bucket_count, total_target_count, total_target_count / total_bucket_count])
            bucket_info.append(["", ""])
        else:
            result[col] = None  # Ignore non-numerical columns

    columns = ["Bucket Range", "Bucket Count", "Target Count", "Target Ratio"]
    df_bucket_info = pd.DataFrame(bucket_info, columns=columns)
    return df_bucket_info


def single_correl_cat(df, radio_array, min_count, n, target_col, df_bucket_info):
    result = []
    for col in df.columns:
        total_count = 0
        total_sum = 0
        if (radio_array.get(col, "") == "categorical" or (
                radio_array.get(col, "") == "numerical" and df[col].nunique() <= n)) and col != target_col:

            # to get <blank> shown separately
            df[col] = df[col].fillna("<blank>")

            grouped = df.groupby(col)[target_col].agg(['sum', 'count'])
            grouped['average'] = grouped['sum'] / grouped['count']
            grouped = grouped.reset_index().rename(
                columns={col: 'column', 'sum': 'sum_of_ones', 'count': 'total_count'})

            # Group expressions with count < min_count into "other" category
            if min_count > 0 and radio_array.get(col, "") == "categorical":
                other_group = grouped[grouped['total_count']
                                      < min_count].copy()
                other_group['column'] = 'other'
                other_group = other_group.groupby('column').sum().reset_index()
                grouped = pd.concat(
                    [grouped[grouped['total_count'] >= min_count], other_group], ignore_index=True)

            # Add a new column to sort by "Other" groups last
            grouped['sort_order'] = grouped['column'].apply(
                lambda x: 1 if x == 'other' else 0)

            # Sort by total count and sort order
            if radio_array.get(col, "") == "categorical":
                grouped = grouped.sort_values(
                    by=['sort_order', 'total_count'], ascending=[True, False])
            else:
                grouped = grouped.sort_values(
                    by=['sort_order', 'column'], ascending=[True, True])
            grouped = grouped.drop('sort_order', axis=1)

            # Append column name as new row
            result.append(
                pd.DataFrame([[col, '', '', '']], columns=['column', 'total_count', 'sum_of_ones', 'average']))

            # Append grouped dataframe
            result.append(
                grouped[['column', 'total_count', 'sum_of_ones', 'average']])

            # Append total row
            total_count += grouped['total_count'].sum()
            total_sum += grouped['sum_of_ones'].sum()
            total_average = total_sum / total_count

            result.append(
                pd.DataFrame([['Total', total_count, total_sum, total_average]],
                             columns=['column', 'total_count', 'sum_of_ones', 'average']))

            # Append a blank row
            result.append(
                pd.DataFrame([['', '', '', '']], columns=['column', 'total_count', 'sum_of_ones', 'average']))

    # Concatenate all results and save to CSV
    result_df = pd.concat(result, axis=0)
    result_df = pd.concat([result_df, df_bucket_info], ignore_index=True)
    result_df.to_csv(save_csv_single_correl, index=False)


def train_tree_classifier(classifier, get_params=True, all_classification=False):
    global df, label_target_var, param_dict, show_ROC

    # find the used algorithm
    if classifier == RandomForestClassifier:
        classifier_name = "Random_Forest"
    elif classifier == ExtraTreesClassifier:
        classifier_name = "Extra_Trees"
    elif classifier == AdaBoostClassifier:
        classifier_name = "AdaBoost"
    elif classifier == GradientBoostingClassifier:
        classifier_name = "Gradient_Boosting"
    elif classifier == xgb.XGBClassifier:
        classifier_name = "XGBoost"
    else:
        classifier_name = "unknown"

    print("Training", classifier_name, "classifier...")

    nan_replacement(df, -999999)
    while not label_target_var:
        set_target_var()

    y = df[label_target_var]
    x = remove_unwanted_datatypes(df)

    if get_params:
        param_dict = get_tree_params()

    number_of_trees = int(param_dict['number of trees'])
    max_depth_of_tree = int(param_dict['max depth of tree'])
    min_samples_to_split_node = int(param_dict['min samples to split node'])
    min_samples_to_be_in_leaf = int(param_dict['min samples to be in leaf'])
    train_split = param_dict['share training set']
    confusion_matrix_cutoff = param_dict['confusion matrix cutoff']
    show_feature_importance = param_dict['show feature importance']
    show_ROC = param_dict['show ROC']
    draw_and_save_trees = param_dict['draw and save trees']
    save_data = param_dict['save the data']
    save_model = param_dict['save the model']

    if train_split < 1:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_split, random_state=rnd_state)
    else:
        x_train = x
        x_test = x
        y_train = y
        y_test = y

    # Create a DataFrame to store the results
    results_df = pd.DataFrame()

    # Add columns for number of observations, set (train or test), target variable, and model score
    results_df['Observation'] = list(range(len(x_train) + len(x_test)))
    results_df['Set'] = ['Train'] * len(x_train) + ['Test'] * len(x_test)
    results_df['Target'] = list(y_train) + list(y_test)
    results_df['Model Score'] = 0.0

    if classifier == AdaBoostClassifier:  # exclude parameters max_depth, min_samples_split, min_samples_leaf
        clf = classifier(random_state=rnd_state, n_estimators=number_of_trees)
    else:
        clf = classifier(random_state=rnd_state, n_estimators=number_of_trees, max_depth=max_depth_of_tree,
                         min_samples_split=min_samples_to_split_node, min_samples_leaf=min_samples_to_be_in_leaf)

    clf.fit(x_train, y_train)

    # Get the predicted probabilities for the positive class (index 1)
    # Probability of positive class for training set
    train_predicted_probs = clf.predict_proba(x_train)[:, 1]
    # Probability of positive class for test set
    test_predicted_probs = clf.predict_proba(x_test)[:, 1]

    # Assign the predicted values to the 'Model Score' column of the DataFrame
    results_df['Model Score'] = np.concatenate(
        (train_predicted_probs, test_predicted_probs))

    # Save the DataFrame to a CSV file
    results_df.to_csv('model_results.csv', index=False)

    if save_data:
        x_train.to_csv(save_csv_name, index=False)

    if show_feature_importance:
        # Get feature importance and sort them in descending order
        importance = clf.feature_importances_
        indices = np.argsort(importance)[::-1]

        # Plot the feature importance
        plt.figure()
        plt.title("Feature importance")
        plt.bar(range(x.shape[1]), importance[indices],
                color="r", align="center")
        plt.xticks(range(x.shape[1]), x.columns[indices], rotation=90)
        plt.xlim([-1, x.shape[1]])
        plt.show()

    # Make predictions on the testing set - prediction method provides probabilities, not binary predictions
    y_pred_proba = clf.predict_proba(x_test)[::, 1]
    y_pred = (y_pred_proba > confusion_matrix_cutoff).astype('float')

    # Make predictions on the training set
    y_train_pred_proba = clf.predict_proba(x_train)[::, 1]
    y_train_pred = (y_train_pred_proba >
                    confusion_matrix_cutoff).astype('float')

    # Evaluate the performance of the model
    print("Accuracy Training Set", metrics.accuracy_score(y_train, y_train_pred))
    if train_split < 1:
        print("Accuracy Test Set", metrics.accuracy_score(y_test, y_pred))

    crrfc = metrics.classification_report(y_test, y_pred)
    print(crrfc)
    # if train_split = 1 it shows the confusion matrix for the training set, otherwise the test set
    if train_split == 1:
        print("Confusion Matrix Training Set:")
    else:
        print("Confusion Matrix Test Set:")
    cmrfc = metrics.confusion_matrix(y_test, y_pred)
    print(cmrfc)

    if train_split == 1 or not all_classification:
        # Calculate ROC metrics for training set
        fpr_train, tpr_train, _ = metrics.roc_curve(
            y_train, y_train_pred_proba)
        auc_train = metrics.roc_auc_score(y_train, y_train_pred_proba)
        auc_dict[classifier_name + "_Training"] = round(auc_train, 4)
        fpr_dict[classifier_name + "_Training"] = fpr_train
        tpr_dict[classifier_name + "_Training"] = tpr_train

    if train_split < 1:
        # Calculate ROC metrics for test set
        fpr_test, tpr_test, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc_test = metrics.roc_auc_score(y_test, y_pred_proba)
        auc_dict[classifier_name + "_Test"] = round(auc_test, 4)
        fpr_dict[classifier_name + "_Test"] = fpr_test
        tpr_dict[classifier_name + "_Test"] = tpr_test

    if draw_and_save_trees:
        fig = plt.figure(figsize=(30, 20))

        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html

        for i in range(number_trees_pics):
            plot_tree(clf.estimators_[i],
                      feature_names=x_train.columns, max_depth=trees_pics_depth,
                      filled=True, impurity=True,
                      rounded=True)
            fig.savefig('figure_name' + str(i) + '.png')

    if save_model:
        joblib.dump(clf, "./" + classifier_name.lower() + "_" +
                    datetime.now().strftime("%Y%m%d_%H%M%S") + ".joblib")


def lr_stepwise_backward_selection(x_train, y_train, p_value_cutoff):
    remaining_features = x_train.columns.tolist()
    while len(remaining_features) > 0:
        x_train_subset = x_train[remaining_features]

        # Create and train the logistic regression model using statsmodels
        logit_model = sm.Logit(y_train, sm.add_constant(x_train_subset))
        result = logit_model.fit()

        # Get p-values for coefficients
        p_values = result.pvalues[1:]

        # Find features with p-values greater than the cutoff
        features_to_remove = p_values[p_values > p_value_cutoff].index.tolist()
        print("Removed features in this step: ", features_to_remove)

        # Stop if no feature needs to be removed
        if not features_to_remove:
            break

        # Remove features with p-values greater than the cutoff
        remaining_features = [
            f for f in remaining_features if f not in features_to_remove]

    return remaining_features, result


def train_logistic_regression(all_classification=False):
    global df, label_target_var, show_ROC

    def drop_cols(data, target_col, collinear_threshold=0.65, insignificance_threshold=0.00):

        # compute the correlation matrix
        # corr_matrix = data.corr(numeric_only=True)
        corr_matrix = data.corr(numeric_only=False)

        print(corr_matrix)

        # create a set to store the columns to drop
        cols_to_drop = set()

        # iterate over the columns
        for col in corr_matrix.columns:
            if col == target_col:
                continue
            # check if the column correlates strongly with another column
            strong_corr = (corr_matrix[col][:-1].abs()
                           > collinear_threshold).sum() > 1
            if strong_corr:
                # check if the column correlates more weakly with the target variable
                other_cols = corr_matrix[abs(
                    corr_matrix[col]) > collinear_threshold].index.tolist()
                other_cols.remove(col)
                max_corr = corr_matrix.loc[other_cols, target_col].abs().max()
                if max_corr >= abs(corr_matrix.loc[col, target_col]):
                    # if max_corr is equal to the correlation between the column and target variable
                    # choose the column with the higher index to drop
                    if abs(max_corr - abs(corr_matrix.loc[col, target_col])) <= 1e-10:
                        drop_col = max(
                            col, corr_matrix.loc[other_cols, target_col].abs().idxmax())
                    else:
                        drop_col = col
                    cols_to_drop.add(drop_col)

            # check if the column correlates weakly with the target variable
            corr_with_target = corr_matrix.loc[col, target_col]
            # print(f'corr_with_target= {corr_with_target}')
            # print(f'abs(corr_with_target)= {abs(corr_with_target)}')
            # print(f'insignificance_threshold= {insignificance_threshold}')
            if abs(corr_with_target) < insignificance_threshold:
                cols_to_drop.add(col)

        # drop the columns and return the updated DataFrame
        return data.drop(cols_to_drop, axis=1)

    nan_replacement(df, -999999)
    while not label_target_var:
        set_target_var()

    param_dict = get_lr_params()
    num_iter = int(param_dict['number of iterations'])
    collinearity = param_dict['collinearity threshold']
    insignificance = param_dict['insignificance threshold']
    train_split = param_dict['share training set']
    confusion_matrix_cutoff = param_dict['confusion matrix cutoff']
    stepwise_backward = param_dict['auto feature elimination']
    p_value_cutoff = param_dict['p-value cutoff']
    show_ROC = param_dict['show ROC']
    save_data = param_dict['save the data']
    save_model = param_dict['save the model']

    y = df[label_target_var]
    # print('Khalid', df)
    x = remove_unwanted_datatypes(df)
    # x=df

    if train_split < 1:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_split, random_state=rnd_state)
    else:
        x_train = x
        x_test = x
        y_train = y
        y_test = y

    # bring the target variable in for correl matrix
    train_data_incl_target = pd.concat([x_train, y_train], axis=1)

    x_train = drop_cols(train_data_incl_target, label_target_var, collinearity, insignificance).drop(label_target_var,
                                                                                                     axis=1)  # drop it again
    # print(x_train)

    # drop the same columns from the test set, i.e. retain only those which are still in the training set
    common_cols_train_test = x_train.columns.intersection(x_test.columns)
    x_test = x_test[common_cols_train_test]

    # Create and train the logistic regression model using statsmodels
    logit_model = sm.Logit(y_train, sm.add_constant(x_train))
    result = logit_model.fit(maxiter=num_iter)

    # Perform stepwise backward selection if enabled
    if stepwise_backward:
        selected_features, result = lr_stepwise_backward_selection(
            x_train, y_train, p_value_cutoff)
        # Update the training and testing data with the selected features
        x_train = x_train[selected_features]
        x_test = x_test[selected_features]

        # Create and train the logistic regression model using statsmodels
        logit_model = sm.Logit(y_train, sm.add_constant(x_train))
        result = logit_model.fit(maxiter=num_iter)

    if save_data:
        x_train.to_csv(save_csv_name, index=False)

    # Get the coefficients and intercept
    coef = result.params[1:]  # Exclude the intercept
    intercept = result.params[0]

    # Make predictions on the testing set - prediction method provides probabilities, not binary predictions
    y_pred_proba = result.predict(sm.add_constant(x_test))
    y_pred = (y_pred_proba > confusion_matrix_cutoff).astype('float')

    # Make predictions on the training set
    y_train_pred_proba = result.predict(sm.add_constant(x_train))
    y_train_pred = (y_train_pred_proba >
                    confusion_matrix_cutoff).astype('float')

    # Evaluate the performance of the model
    print("Accuracy Training Set", metrics.accuracy_score(y_train, y_train_pred))
    if train_split < 1:
        print("Accuracy Test Set", metrics.accuracy_score(y_test, y_pred))

    # Calculate the standard errors for coefficients and the intercept
    std_err = result.bse[1:]
    intercept_se = result.bse[0]

    # Calculate the p-values for coefficients and the intercept
    p_values = result.pvalues[1:]
    intercept_p_value = result.pvalues[0]

    # Create a list of tuples containing the feature names, coefficients, standard errors, and p-values
    coef_list = list(zip(x_train.columns, coef, std_err, p_values))

    # Insert the intercept at the beginning of the coef_list
    coef_list.insert(0, ("Intercept", intercept,
                     intercept_se, intercept_p_value))

    # Make a Connection to the Reporting Database (SQl SERVER Managment)
    conn = pymssql.connect(server='T24DB-REPORTS'                           # ,user='zeppelin'
                           # ,password='zeppelin'
                           , database='His_T24_Reports')
    cursor = conn.cursor(as_dict=True)
    # Delete all The Data from the table Every Run time for the module
    cursor.execute("Delete His_T24_Reports.dbo.CorrlectionsValue")
    # Insert into the table the (Feature,Coefficient,Standard Error,P-value)
    for i in range(len(coef_list)):
        cursor.executemany(
            "INSERT  INTO  His_T24_Reports.dbo.CorrlectionsValue VALUES (%s,%d,%d,%d)", [(
                coef_list[i][0], coef_list[i][1], coef_list[i][2], coef_list[i][3])]
        )
    conn.commit()
    # Close the Connection
    conn.close()

    # Print the coefficients, intercept, standard errors, and p-values in a table format
    headers = ["Feature", "Coefficient", "Standard Error", "p-value"]
    table = tabulate(coef_list, headers=headers, tablefmt="fancy_grid")
    print(table)

    crrfc = metrics.classification_report(y_test, y_pred)
    print(crrfc)
    # if train_split = 1 it shows the confusion matrix for the training set, otherwise the test set
    if train_split == 1:
        print("Confusion Matrix Training Set:")
    else:
        print("Confusion Matrix Test Set:")
    cmrfc = metrics.confusion_matrix(y_test, y_pred)
    print(cmrfc)

    if train_split == 1 or not all_classification:
        # Calculate ROC metrics for training set
        fpr_train, tpr_train, _ = metrics.roc_curve(
            y_train, y_train_pred_proba)
        auc_train = metrics.roc_auc_score(y_train, y_train_pred_proba)
        auc_dict["Logistic_Regression_Training"] = round(auc_train, 4)
        fpr_dict["Logistic_Regression_Training"] = fpr_train
        tpr_dict["Logistic_Regression_Training"] = tpr_train

    if train_split < 1:
        # Calculate ROC metrics for test set
        fpr_test, tpr_test, _ = metrics.roc_curve(y_test, y_pred_proba)
        auc_test = metrics.roc_auc_score(y_test, y_pred_proba)
        auc_dict["Logistic_Regression_Test"] = round(auc_test, 4)
        fpr_dict["Logistic_Regression_Test"] = fpr_test
        tpr_dict["Logistic_Regression_Test"] = tpr_test

    if save_model:
        joblib.dump(result, "./logistic_regression_" +
                    datetime.now().strftime("%Y%m%d_%H%M%S") + ".joblib")


def train_kMeans():
    df_kMeans = df
    headers = df_kMeans.columns

    param_dict = get_kMeans_params()

    calc_elbow = param_dict['show elbow chart']
    min_groups = int(param_dict['min groups'])
    max_groups = int(param_dict['max groups'])
    number_init = int(param_dict['number random initiations'])
    save_data = param_dict['save the data']
    save_model = param_dict['save the model']

    df_kMeans = remove_unwanted_datatypes(df_kMeans)

    # Normalize the continuous variables >> perhaps make this optional
    scaler = StandardScaler()
    # unfortunately this converts the df object into a numpy array
    df_kMeans = scaler.fit_transform(df_kMeans)

    # Fit KMeans model to the data
    if calc_elbow:
        inertias = []
        for num_clusters in range(min_groups, max_groups + 1):
            kmeans = KMeans(n_clusters=num_clusters,
                            n_init=number_init, random_state=1).fit(df_kMeans)
            inertias.append(kmeans.inertia_)

        # Plot elbow chart
        plt.plot(range(min_groups, max_groups + 1), inertias)
        plt.title('Elbow chart')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

        # Choose the number of clusters you want to create
        num_clusters = get_input("the number of clusters")
    else:
        num_clusters = min_groups

    # Create a k-means object
    kmeans = KMeans(n_clusters=num_clusters, n_init=number_init,
                    random_state=1).fit(df_kMeans)

    # Add the cluster labels to the numpy array
    labels = (kmeans.labels_ + 1).reshape(-1, 1)
    result = np.hstack((df_kMeans, labels))
    df_kMeans = pd.DataFrame(result, columns=list(headers) + ['cluster'])

    # measures how well each point fits into its assigned cluster, based on the distance between the point and the
    # neighboring clusters. A higher silhouette score indicates that the point is well-clustered, while a lower score
    # suggests that the point may belong to a different cluster. >> between -1 and 1
    siL_score = metrics.silhouette_score(
        df_kMeans.drop('cluster', axis=1), labels.ravel())

    # measures the ratio of between-cluster variance to within-cluster variance. A higher Calinski-Harabasz index
    # indicates better-defined clusters.
    c_h_score = metrics.calinski_harabasz_score(
        df_kMeans.drop('cluster', axis=1), labels.ravel())

    # average similarity between each cluster and its most similar
    # cluster, relative to the average dissimilarity between each cluster and its least similar cluster
    d_b_score = metrics.davies_bouldin_score(
        df_kMeans.drop('cluster', axis=1), labels.ravel())

    print(f"Silhouette Score: {siL_score}")  # >0.5
    print(f"Calinski-Harabasz Score: {c_h_score}")  # >100
    print(f"Davies-Bouldin Score: {d_b_score}")  # <1.0

    # Retrieve the cluster centers
    centers = kmeans.cluster_centers_

    df_centers = pd.DataFrame(centers, columns=list(headers))
    df_centers['center'] = ['Center{}'.format(
        i) for i in range(1, kmeans.n_clusters + 1)]

    # Add the centers to the dataframe with their respective cluster names
    df_kMeans = pd.concat([df_kMeans, df_centers], axis=0, ignore_index=True)

    df_kMeans = df_kMeans.sort_values(by='center', ascending=True)

    # Replace the empty "cluster" values for the centers with the corresponding "center" value
    df_kMeans['cluster'].fillna(df_kMeans['center'], inplace=True)

    df_kMeans = df_kMeans.drop('center', axis=1)

    messagebox.showinfo("k-means", "Model trained.")

    # This code calculates the cosine similarity between the i-th and j-th rows of the X dataframe. The resulting
    # cos_sim value will range between -1 and 1, with higher values indicating greater similarity. and i and j are
    # the indices of the two observations
    # cos_sim = cosine_similarity(df_kMeans.iloc[[5]], df_kMeans.iloc[[6]])[0][0]
    # print(f"cosine_similarity: {cos_sim}")

    # This code calculates the Euclidean distance between the i-th and j-th rows of the X dataframe. The resulting
    # euclidean_dist value will be non-negative, with lower values indicating greater similarity.
    # euclidean_dist = euclidean_distances(df_kMeans.iloc[[5]], df_kMeans.iloc[[6]])[0][0]
    # print(f"euclidean_dist: {euclidean_dist}")

    if save_data:
        df_kMeans.to_csv("kMeans_modeltest2.csv", index=False)
        messagebox.showinfo("k-means", "Data saved.")

    if save_model:
        joblib.dump(kmeans, 'kmeans_model.sav')

    # load the saved model from disk
    #     # load the saved k-means model and scaler object from disk
    #     kmeans_model = joblib.load('kmeans_model.sav')
    #     scaler = joblib.load('scaler.sav') >> to be saved
    #
    #     # load the new data
    #     new_data = pd.read_csv('new_data.csv')
    #
    #     # apply the same scaling used for the original data
    #     scaled_data = scaler.transform(new_data)
    #
    #     # use the k-means model to predict cluster labels for the new data
    #     predicted_labels = kmeans_model.predict(scaled_data)


def apply_model():
    global df
    # Open a file dialog to select the model file
    file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[
                                           ("Joblib files", "*.joblib")])

    print(file_path)

    if file_path:
        # Load the selected model
        loaded_model = joblib.load(file_path)
        # Check the class type of the loaded model
        model_class = type(loaded_model)
        print(model_class)
        messagebox.showinfo("Information", "Model loaded. Gets applied...")

        # checks if LogReg model (actually checks if trained with statsmodels)
        if hasattr(loaded_model, "params"):
            print(loaded_model.params)

            # add a column with 1.0 for the constant (intercept)
            df = sm.add_constant(df)

            print(df)

            # Make predictions on the new data
            new_predictions = loaded_model.predict(df)

            # Add the predicted probabilities as a new column to the DataFrame
            df['Prediction'] = new_predictions

            print(df)

        else:
            # Extract the feature names used during training
            original_feature_columns = loaded_model.feature_names_in_

            print(f"Original features columns: {original_feature_columns}")

            # Filter the DataFrame columns to retain (and sort) only those used during model training
            df = df[original_feature_columns]

            class1_prob = loaded_model.predict_proba(
                df)[:, 1]  # Extract probability of class 1

            # Create a new column for the probability of class 1
            df["Predicition"] = class1_prob

            print(df)

        messagebox.showinfo("Information", "Model applied. Scores calculated.")


def if_comparison3():
    global df
    connector = "x"
    new_column_name = ""

    while not (connector == "="):
        if connector == "x":
            first_column, new_column_name = if_comparison2()

        connector = button_menu("Choose the connector.", connector_array)

        if connector == "Cancel":
            return

        if not (connector == "="):
            second_column, second_col_name = if_comparison2()
            new_column_name = new_column_name + "__" + connector + "__" + second_col_name
            if connector == "AND":
                first_column = first_column * second_column
            elif connector == "OR":
                first_column = min(first_column + second_column, 1)

    df.insert(len(df.columns), new_column_name, first_column)

    messagebox.showinfo("Column Operations",
                        f"A new column {new_column_name} was created.")


def if_comparison2():
    global df
    sel_col_label = button_menu("Select a column.", df.columns.tolist())
    sel_col = df[sel_col_label]

    num_unique = sel_col.nunique()

    show_threshold = top_unique_value_threshold

    if sel_col.dtype == object or num_unique <= 5:

        if num_unique > show_threshold:
            show_threshold = get_input(
                "the number of top value counts to be shown")

        # Get the unique values and their counts in the column
        value_counts = sel_col.value_counts()

        # Get the top values based on count of appearance
        unique_values = value_counts.index.tolist()[:show_threshold]

        sel_expressions = checkbox_menu(
            "Select the expressions.", unique_values)

        if len(sel_expressions) == 1:
            or_equal = "="
        else:
            or_equal = "OR"

        logic = button_menu("Choose the logic.", [or_equal, "NOT"])

        new_col_name = "multiple"
        if len(sel_expressions) < 5:
            new_col_name = '_'.join(str(expr) for expr in sel_expressions)

        if logic == or_equal:
            new_column = df[sel_col_label].isin(sel_expressions).astype(int)
        else:
            new_column = (~df[sel_col_label].isin(sel_expressions)).astype(int)

        new_col_name = sel_col_label + "_" + logic + "_" + new_col_name

    else:
        # Get summary statistics for the numeric column
        summary_stats = [np.min(sel_col), np.max(sel_col), np.mean(sel_col), np.median(sel_col),
                         np.std(sel_col)]
        comp_choice, value1, value2 = math_comp_menu(
            "Choose the logic.", summary_stats)
        value1 = float(value1)
        if value2 is not None:
            value2 = float(value2)

            if value2 < value1:
                value_temp = value2
                value2 = value1
                value1 = value_temp

        if comp_choice == "between":
            new_column = ((sel_col >= float(value1)) & (
                sel_col <= float(value2))).astype(int)
        elif comp_choice == "not between":
            new_column = (
                ~(sel_col.between(float(value1), float(value2)))).astype(int)
        elif comp_choice == "=":
            new_column = (sel_col == float(value1)).astype(int)
        elif comp_choice == "!=":
            new_column = (sel_col != float(value1)).astype(int)
        elif comp_choice == "<":
            new_column = (sel_col < float(value1)).astype(int)
        elif comp_choice == "<=":
            new_column = (sel_col <= float(value1)).astype(int)
        elif comp_choice == ">":
            new_column = (sel_col > float(value1)).astype(int)
        elif comp_choice == ">=":
            new_column = (sel_col >= float(value1)).astype(int)

        new_col_name = sel_col_label + "_" + comp_choice + "_" + str(value1)
        if value2 is not None:
            new_col_name = new_col_name + "_" + str(value2)

        # messagebox.showinfo("New Column", f"Column {new_col_name} has been created.")

    return new_column, new_col_name


def test():
    return


def test2():
    return


main_menu()
