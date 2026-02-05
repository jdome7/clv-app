import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from streamlit_agraph import agraph, Node, Edge, Config

@st.cache_resource
def get_connection():
    return sqlite3.connect('deploy_clv.db', check_same_thread=False)


conn = get_connection()

@st.cache_data
def get_all_ids(_conn):
    return pd.read_sql("SELECT customer_id FROM customer_profiles ORDER BY customer_id", _conn)['customer_id'].tolist()

all_ids = get_all_ids(conn)

if "selected_customer_id" not in st.session_state:
    st.session_state.selected_customer_id = all_ids[0] # Default to the first customer

def on_change_callback():
    st.session_state.selected_customer_id = st.session_state.customer_selector


selected_id = st.selectbox(
    "Search Customer ID",
    options=all_ids,
    index=all_ids.index(st.session_state.selected_customer_id),
    key="customer_selector",
    on_change=on_change_callback
)

selected_id = st.session_state.selected_customer_id

def get_structural_twins(customer_id, sim_index, conn, n=3):
    query = f"""
    SELECT customer_id, behavioral_similarity_index, clv, expected_monetary_value_segment
    FROM customer_profiles 
    WHERE customer_id != {customer_id}
    ORDER BY ABS(behavioral_similarity_index - {sim_index}) ASC
    LIMIT {n}
    """
    return pd.read_sql(query, conn)


def show_agraph_network(customer_id, conn):
\
    query = f"""
    SELECT DISTINCT e.*, p.Description, p.pagerank_norm, e.total_spent, e.last_purchase_date
    FROM customer_edges e
    JOIN products p ON e.StockCode = p.StockCode
    WHERE e.customer_id = {customer_id}
    """
    edges_df = pd.read_sql(query, conn)
    
    nodes = []
    edges = []
    
 
    nodes.append(Node(id=str(customer_id), 
                      label=f"Customer {customer_id}", 
                      font={'color': "#FFFFFF"},
                      size=30, 
                      color="#25347A", 
                      shape="circle"))
    
 
    for _, row in edges_df.iterrows():

        clean_desc = str(row['Description']).strip()

        pr_val = row['pagerank_norm'].iloc[0]
        
        node_size = 15 + (pr_val * 25)

        product_id = row['StockCode']

        total_spent = str(row['total_spent'].iloc[0]).strip()
        last_purchase_date = str(row['last_purchase_date'].iloc[0]).strip()
        purchase_count = str(row['purchase_count']).strip()

        
        nodes.append(Node(id=row['StockCode'], 
                          label=product_id,
                          font={'color': "#FFFFFF"},
                          title=f"Item: {clean_desc}\nPurchase Count: {purchase_count}\nTotal Spent: ${total_spent}\nLast Purchase Date: {last_purchase_date}",
                          size=node_size, 
                          color="#1C83E1",
                          shape="circle"))
        

        edges.append(Edge(source=str(customer_id), 
                          target=row['StockCode'], 
                          color="#CFE7F0",
                          width=int(row['purchase_count'])))

    config = Config(
            width='100%', 
            height=500, 
            directed=False, 
            physics=True, 
            hierarchical=False,
            nodeHighlightBehavior=True, 
            highlightColor="#F7A7A6",
            collapsible=False, 
            fit=True,             
            initialZoom=1,       
            minZoom=1,            
            maxZoom=1,            
            staticGraphWithDragAndDrop=True 
        )

    return agraph(nodes=nodes, edges=edges, config=config)



def get_recommendations(customer_id, conn):
    query = f"""
    WITH MyProducts AS (
        SELECT StockCode FROM customer_edges WHERE customer_id = {customer_id}
    ),
    Neighbors AS (
        SELECT DISTINCT e2.customer_id
        FROM customer_edges e2
        JOIN MyProducts mp ON e2.StockCode = mp.StockCode
        WHERE e2.customer_id != {customer_id}
        LIMIT 1500 -- Optimization: We only need a sample of similar people to get great results
    )
    SELECT p.Description, COUNT(e3.customer_id) as link_strength
    FROM customer_edges e3
    JOIN Neighbors n ON e3.customer_id = n.customer_id
    JOIN products p ON e3.StockCode = p.StockCode
    WHERE e3.StockCode NOT IN (SELECT StockCode FROM MyProducts)
    GROUP BY p.Description
    ORDER BY link_strength DESC
    LIMIT 3
    """
    return pd.read_sql(query, conn)


def behavioral_galaxy_3d(conn, min_p=0.0, max_p=1.0):
    query = f"""
    SELECT *
    FROM customer_profiles
    WHERE probability_alive BETWEEN {min_p} AND {max_p}
    """
    summary = pd.read_sql(query, conn)

    features = ['clv', 'probability_alive', 'pagerank', 'centrality', 
                'unique_products', 'behavioral_similarity_index', 'product_concentration']

    X_subset = summary[features].fillna(0)

 
    X_scaled = StandardScaler().fit_transform(X_subset)

    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
    coords_3d = tsne_3d.fit_transform(X_scaled)


    summary['x_3d'] = coords_3d[:, 0]
    summary['y_3d'] = coords_3d[:, 1]
    summary['z_3d'] = coords_3d[:, 2]

    st.subheader("Customer Embeddings")
    st.caption("A 3D projection of customer behavior. Each dot is a customer; proximity represents behavioral similarity.")

    df_3d = summary.copy()


    if df_3d.empty:
        st.warning("No customer profiles found.")
        return

    fig = px.scatter_3d(
        df_3d, 
        x='x_3d', y='y_3d', z='z_3d',
        color= 'expected_monetary_value_segment',
        size='clv', # Bigger dots for higher value
        size_max=15,
        opacity= 1,
        hover_data=['customer_id', 'clv', 'probability_alive'],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark" # Dark mode makes the 'galaxy' pop
    )
    
    # Adjusting the layout for a cleaner look
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

    #     # Add this inside your function logic
    # df_3d['Risk_Level'] = pd.cut(df_3d['probability_alive'], 
    #                             bins=[0, 0.3, 0.7, 1.0], 
    #                             labels=['High Risk', 'Neutral', 'Healthy'])

    # # Then change the px.scatter_3d color parameter:
    # color='Risk_Level'




def plot_pareto_curve(conn):
    st.subheader("% of Total Revenue vs. % of Customer Base")
    

    pareto_df = pd.read_sql("SELECT * FROM pareto_curve", conn)
    
    fig = px.line(
        pareto_df, 
        x='perc_customers', 
        y=['actual_revenue_cum', 'predicted_revenue_cum'],
        labels={
            'perc_customers': '% of Customer Base',
            'value': '% of Total Revenue',
            'variable': 'Source'
        },
        title="Actual vs. Predicted Revenue Concentration",
        color_discrete_map={
            'actual_revenue_cum': '#979797',  
            'predicted_revenue_cum': '#FF4B4B' 
        }
    )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

def feature_importance(conn):
    st.subheader("📊 Feature Importance")
    fi_df = pd.read_sql("SELECT * FROM importance_df", conn)
    #st.write(fi_df)
    st.dataframe(fi_df, use_container_width=True)

def model_performance(conn):
    st.subheader("Model Perfromance")
    mi_df = pd.read_sql("SELECT * FROM model_performance", conn)
    #st.write(mi_df)
    st.dataframe(mi_df, use_container_width=True)

def customer_tab(selected_id):
    conn = get_connection()
    st.header("👤 Customer Profile")
    

    # Add a sort option, add last purchase date, add links to similar customers


    if selected_id:
        profile = pd.read_sql(f"SELECT * FROM customer_profiles WHERE customer_id = {selected_id}", conn).iloc[0]
        g_stats = pd.read_sql("SELECT * FROM global_stats", conn).iloc[0]
        
        #Predicted at top
        clv_delta = profile['clv'] - profile['clv_last_6_months']

        # 2. Format the string for the label
        delta_label = f"${abs(clv_delta):,.2f} vs Avg"

        col1, col2, col3, col4 = st.columns(4)
  
        col1.metric(
            label="Predicted Revenue (6 Months)", 
            value=f"${profile['clv']:,.2f}", 
            delta=f"{clv_delta:,.2f} vs Last 6 Months",
            delta_color="normal"
        )

        col2.metric("Probability Active", f"{profile['probability_alive']:.2%}")
        col3.metric("Exp. Avg. Purchase $ Tier", profile['expected_monetary_value_segment'])
        col4.metric("Exp. Purchase Freq Tier", profile['expected_frequency_segment'])

        st.divider()

        left_col, right_col = st.columns([2, 1])
        with left_col:
            st.subheader("Purchase Network")
            show_agraph_network(selected_id, conn)
    
        #hover question marks things
        with right_col:
            st.subheader("Similar Customers")
            twins = get_structural_twins(selected_id, profile['behavioral_similarity_index'], conn)
            
            # for _, twin in twins.iterrows():
            #     st.info(f"ID: {int(twin['customer_id'])} | CLV: ${twin['clv']:,.0f}")
            for _, twin in twins.iterrows():
                twin_id = int(twin['customer_id'])
                
                # Create a button for each twin
                if st.button(f"Customer {twin_id} | CLV: ${twin['clv']:,.0f}", key=f"btn_{twin_id}"):
                    st.session_state.selected_customer_id = twin_id
                    st.rerun() # This force-refreshes the app to the new ID's profile

            st.subheader("Recommended Products")
            recs = get_recommendations(selected_id, conn)
            if not recs.empty:
                for i, row in recs.iterrows():
                    st.success(f"{row['Description']}")
            
            
        st.divider()

        st.subheader("Last 6 Months")
        col1, col2, col3 = st.columns(3)
        col1.metric("Revenue", f"${profile['clv_last_6_months']:,.2f}")
        col2.metric("No. Purchases", f"{profile['purchases_last_6_months']}")
        col3.metric("Average Purchase Amount", f"${profile['monetary_value_last_6_months']:,.2f}")
        
        st.divider()

        st.subheader("Transaction History")

        history = pd.read_sql(f"SELECT InvoiceDate, Description, Quantity, Price, Revenue FROM transactions WHERE customer_id = {selected_id} ORDER BY InvoiceDate DESC", conn)
        st.dataframe(history, use_container_width=True)
        st.write("**Since 12/1/2009*")


def business_tab():

    conn = get_connection()
    st.header("Business Overview")


    g_stats = pd.read_sql("SELECT * FROM global_stats", conn).iloc[0]
    total_customers = pd.read_sql("SELECT COUNT(*) as total_customers FROM customer_profiles", conn).iloc[0]['total_customers']
    total_revenue = pd.read_sql("SELECT SUM(clv) as total_revenue FROM customer_profiles", conn).iloc[0]['total_revenue']
    
    main_left, main_right = st.columns([1, 2])

    with main_left:
 
        st.metric("Total Customer Count", f"{total_customers}")
        st.metric("Total Predicted Revenue (6 Months)", f"${total_revenue:,.0f}")
        st.metric("Average Predicted Customer Revenue", f"${g_stats['avg_clv']:,.0f}")
        st.metric("Average Probability Alive", f"{g_stats['avg_prob_alive']:.2%}")

    with main_right:
        plot_pareto_curve(conn)

    st.divider()


    # st.sidebar.header("Galaxy Filters")
    # p_range = st.sidebar.slider(
    #     "Select P(Alive) Range",
    #     0.0, 1.0, (0.0, 1.0), 0.05
    # )

    # metrics_query = f"SELECT COUNT(*) as count, SUM(clv) as val FROM customer_profiles WHERE probability_alive BETWEEN {p_range[0]} AND {p_range[1]}"
    # stats = pd.read_sql(metrics_query, conn).iloc[0]
    
    # col1, col2 = st.columns(2)
    # col1.metric("Selected Customer Count", f"{int(stats['count'])}")
    # col2.metric("Filtered CLV Pool", f"${stats['val']:,.0f}")

    # 2. The Filtered 3D Galaxy
    # behavioral_galaxy_3d(conn, min_p=p_range[0], max_p=p_range[1])
    # with st.expander("View Customer Embeddings"):
    #     #add loading

    #     st.container(behavioral_galaxy_3d(conn,0,1))
    


def methodology_tab():
    conn = get_connection()
    st.header("Methodology")

    st.subheader("Data Source")
    st.write("""
    The data used in this project was sourced from a UK-based company that sells giftware online, 
    with a customer base of mainly wholesalers. The dataset is made up of all of the transactions 
    made between 01/12/2009 and 09/12/2011, with 1,067,371 rows of data containing information 
    such as the customer ID, invoice date, product ID, quantity purchased, and more. 
    
    The full dataset can be found at: https://archive.ics.uci.edu/dataset/502/online+retail+ii
    """)

    st.subheader("Tech Stack")
    st.markdown("""
    * **Core:** Python, XGBoost, Pandas, NumPy, Scikit-Learn
    * **Probabilistic Modeling:** Lifetimes
    * **Networks/Graphing:** NetworkX, Streamlit-AgGraph
    * **Data Management:** SQLite
    * **UI/UX:** Streamlit, Plotly
    """)

    st.subheader("Bipartite Graph Modeling")
    st.write("""
    The standard model for predicting customer value (BG/NBD) treats customers as isolated entities, 
    when we can perhaps better understand and predict customer behavior by creating networks and 
    studying how groups of customers behave. There's a story to be told with the relationships 
    between customers and products, and graphs allow us to identify communities of customers with 
    similar purchasing habits, as well as products that are central to the business’s ecosystem.
    
    To incorporate this information into the XGBoost model, I first constructed a bipartite graph 
    (a graph with two sets of nodes, where an edge can only be formed between nodes of different sets). 
    Customers are one set of nodes, and products are the other set of nodes. The edges connect 
    customers to the products they purchased, and the weights of each edge are the total dollar amount 
    spent on each product. 
    """)

    

    st.write("""
    The degree of a customer can be understood as how many unique products they purchased. From this 
    graph construction, I was able to generate a host of insights and new features to better 
    understand this dataset using the NetworkX Python library.
    
    The most important feature in the final XGBoost was actually a graph-based one, PageRank, which is 
    an algorithm that identifies the importance of a node in a network based on the quality and 
    quantity of nodes connected to it. Customers who score high in the PageRank metric are those who 
    are representative of the broader network, likely buying popular products and shopping across 
    categories. This feature provides more context, defining customer value as not only those who 
    spend a lot, but also those who buy central products and have influence over the ecosystem.
    """)

    st.markdown("### PageRank Algorithm")
    st.latex(r"PR(u) = \frac{1-d}{N} + d \sum_{v \in B_u} \frac{PR(v)}{L(v)}")
    st.write("""
    Where:
    * **PR(u)**: The PageRank of node $u$.
    * **d**: The damping factor (typically 0.85), representing the probability that a user continues following links.
    * **N**: Total number of nodes in the network.
    * **B_u**: The set of nodes connected to node $u$.
    * **L(v)**: The number of outbound links from neighbor $v$.
    """)

    st.write("""
    Some other important insights from this graph were being able to capture if a customer tends to 
    buy 10 different items once or 1 item ten times, which customers were purchasing similar products, 
    and how valuable these similar customers were.
    """)

    st.subheader("Model Evaluation")
    st.write("""
    The baseline used was a BG/NBD probabilistic model, which models customer churn with beta and 
    geometric probability distributions, and purchase frequency with a negative binomial distribution. 
    To evaluate the models, the data was split into calibration and holdout periods in a roughly 
    75/25 split. 
    
    * **Calibration Period:** 12/1/2009 to 6/12/2011 (~580 days)
    * **Holdout Period:** End of calibration to 12/9/2011 (~180 days)
    
    The target variable that the models predicted were the individual customer revenue generated over 
    the holdout period. The XGBoost model outperformed the probabilistic model in every metric I observed, 
    both on an aggregate and individual level as displayed in the table below.
    """)
    model_performance(conn)

    st.subheader("Lessons")
    st.write("""
    XGBoost does not naturally understand time and cannot extrapolate data in the way that 
    probabilistic modeling can. This gave me a challenge moving from the training to inference, as 
    given input data over a larger period of time, the model may interpret the customer as stronger 
    than they actually are due to higher number of purchases, while not properly scaling for the 
    amount of time passed. To solve this, I made the observed time period and time-based features 
    consistent across training and inference. 
    
    With the probabilistic model baseline, I was also able to identify which customers the XGBoost 
    model’s predictions differed greatly from than expected. For customers with strong indicators 
    such as buying central products and spending high amounts on average (similar behavior to top 
    customers) the model may predict a high customer revenue despite a dropping or low probability 
    of being alive. This can be interpreted as optimism bias towards certain customer types.
    """)




if __name__ == "__main__":
    st.set_page_config(page_title="Retail Customer Explorer", layout="wide")
    tab1, tab2, tab3 = st.tabs(["Customer Profile", "Business Overview", "Methodology"])

    with tab1:
        customer_tab(selected_id)
    
    with tab2:
        business_tab()
        
    with tab3:
        methodology_tab()