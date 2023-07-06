#!/usr/bin/env python
# coding: utf-8

# Case Study: Diagnosing and Advising an E-commerce Company

# In[1]:


# 2-2 Data Import

import pandas as pd # Python library for data wrangling and basic analysis

# importing the data as dataframe objects
customers = pd.read_csv("/Users/Jun/Desktop/Olist/olist_customers_dataset.csv")
order_items = pd.read_csv("/Users/Jun/Desktop/Olist/olist_order_items_dataset.csv")
order_reviews = pd.read_csv("/Users/Jun/Desktop/Olist/olist_order_reviews_dataset.csv")
orders = pd.read_csv("/Users/Jun/Desktop/Olist/olist_orders_dataset.csv")
products = pd.read_csv("/Users/Jun/Desktop/Olist/olist_products_dataset.csv")
sellers = pd.read_csv("/Users/Jun/Desktop/Olist/olist_sellers_dataset.csv")
categories = pd.read_csv("/Users/Jun/Desktop/Olist/product_category_name_translation.csv")


# In[6]:


# 2-3 Checking for ROSCC

# enabling display of every output from one cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# checing for reliability (accuracy)
customers.head(3) # no need to check for anything

order_items.head(3)
order_items.describe() # both price & freight_value are made up of reasonable values only

order_reviews.head(3)

# every review_answer_timestamp comes after review_creation_date
sum(order_reviews["review_creation_date"] > order_reviews["review_answer_timestamp"])
orders.head(3)
sum(orders["order_purchase_timestamp"] > orders["order_approved_at"]) # no issue found

# issue found - over 1000 orders approved after being delivered to carrier
sum(orders["order_approved_at"] > orders["order_delivered_carrier_date"])

# issue found likewise
sum(orders["order_delivered_carrier_date"] > orders["order_delivered_customer_date"])
sum(orders["order_purchase_timestamp"] > orders["order_delivered_customer_date"]) # no issue found

products.head(3)
products.describe() # all figures valid; no issue found

sellers.head(3) # no need

categories.head(3) # no need

# checing for sufficiency (completeness)
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(customers.isnull(), cbar = False) # no missing values
plt.show()
sns.heatmap(order_items.isnull(), cbar = False) # no missing values
plt.show()
sns.heatmap(order_reviews.isnull(), cbar = False) # extent of missing values unproblematic
plt.show()
sns.heatmap(orders.isnull(), cbar = False) # extent of missing values unproblematic
plt.show()
sns.heatmap(products.isnull(), cbar = False) # extent of missing values unproblematic
plt.show()
sns.heatmap(sellers.isnull(), cbar = False) # no missing values
plt.show()
sns.heatmap(categories.isnull(), cbar = False) # no missing values
plt.show()


# In[3]:


# 3-1 Merging

# join 1
# creating 'order_items_new'
order_items_new = order_items.groupby(["order_id", "product_id"]).agg({"order_item_id": "count", "seller_id": "max", "price": "sum", "freight_value": "sum"}).reset_index()
order_items_new.columns = ["order_id", "product_id", "quantity", "seller_id", "price", "freight_value"]
order_items_new.head(10) # granularity: every product from every order

orders_new = pd.merge(left = orders, right = order_items_new, how = "left", on = "order_id")
orders_new.head(5) # granularity: every product from every order (no order dropped)

# join 2
orders_new = pd.merge(left = orders_new, right = products, how = "left", on = "product_id")
orders_new.head(5) # granularity: every product from every order (no order dropped)

# join 3
orders_new = pd.merge(left = orders_new, right = categories, how = "left", on = "product_category_name")
orders_new.head(5) # granularity: every product from every order (no order dropped)

# join 4
orders_new = pd.merge(left = orders_new, right = sellers, how = "left", on = "seller_id")
orders_new.head(5) # granularity: every product from every order (no order dropped)

# join 5
orders_new = pd.merge(left = orders_new, right = customers, how = "left", on = "customer_id")
orders_new.head(5) # granularity: every product from every order (no order dropped)

# join 6
# creating 'order_reviews_new'
temp = orders_new.groupby("order_id")["product_id"].count()
single_product_orders = pd.DataFrame(temp[temp == 1].index) # orders that have only one product
single_product_orders["is_single_product_order"] = True
order_reviews_new = pd.merge(left = single_product_orders, right = order_reviews, how = "right", on = "order_id")
order_reviews_new["product_review_score"] = order_reviews_new["review_score"]

import numpy as np
order_reviews_new.loc[order_reviews_new["is_single_product_order"] != True, "product_review_score"] = np.nan
order_reviews_new = order_reviews_new[["order_id", "review_score", "product_review_score"]]
order_reviews_new.rename(columns = {"order_id":"order_id", "review_score":"overall_order_review_score", "product_review_score":"overall_product_review_score"}, inplace = True)
order_reviews_new = order_reviews_new.groupby("order_id")["overall_order_review_score","overall_product_review_score"].agg("mean").reset_index() # granularity: every aggregated review from every order

orders_new = pd.merge(left = orders_new, right = order_reviews_new, how = "left", on = "order_id")
orders_new.head(5) # granularity: every product from every order (no order dropped)

orders_new.drop_duplicates(inplace = True) # ensuring no perfect duplicates


# In[4]:


# 3-2 Final Processing

# feature selection
orders_new.drop(axis = 1, columns = ["product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty"], inplace = True)

# missing values
sns.heatmap(orders_new.isnull(), cbar = False) # heatmap
orders_new[orders_new["product_id"].isnull()] # these orders are all either cancelled or unavailable

# data consistency
orders_new["order_status"] = orders_new["order_status"].map({"unavailable":"unavailable", "delivered":"delivered", "canceled":"canceled", "shipped":"shipped", "created":"unprocessed", "invoiced":"unprocessed", "processing":"unprocessed", "approved": "unprocessed"}) # redefining order_status
# reorganisation of product categories was done in Tableau


# In[30]:


# 4-2 The First Question

# QRR
quarter = pd.to_datetime(orders_new["order_purchase_timestamp"]).dt.to_period("Q")
cuid = orders_new["customer_unique_id"]

def purchased_last_quarter(x):    
    cond = (quarter == x[1]) & (cuid == x[0])
    return cond.any()

last_quarter = quarter - 1
helper = (cuid + " " + last_quarter.astype(str)).str.split(" ")
orders_new["purchased_last_quarter"] = helper.apply(purchased_last_quarter)

orders_new["quarter"] = quarter
numerator = orders_new[orders_new["purchased_last_quarter"] == 1].groupby("quarter")["customer_unique_id"].nunique()
denominator = orders_new.groupby("quarter")["customer_unique_id"].nunique()
numerator / denominator.shift(1) # QRR for each applicable fiscal quarter

