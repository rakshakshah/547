import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel , euclidean_distances
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
from sklearn.metrics import silhouette_score
from IPython.display import display


# Step 1: get all stocks in S&P500 (Wikipedia）
wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(wiki_url)
sp500_df = tables[0]
tickers = sp500_df['Symbol'].tolist()
industries = sp500_df.set_index('Symbol')['GICS Sector'].to_dict()

# Step 2: choose industries
target_sectors = [
    'Information Technology',
    'Consumer Discretionary', 'Consumer Staples',
    'Financials', 'Health Care', 'Energy'
]
liquidity_threshold = 1_000_000
max_companies_to_check = 100  
selected_stocks = []

# Step 3: choose stocks
for ticker in tickers[:max_companies_to_check]:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        sector = industries.get(ticker, "")
        if sector not in target_sectors:
            continue

        market_cap = info.get("marketCap", None)
        avg_volume = info.get("averageDailyVolume10Day", info.get("volume", 0))
        name = info.get("longName", "")

        if not market_cap or avg_volume is None:
            continue
        if avg_volume < liquidity_threshold:
            continue
        if any(keyword in name.lower() for keyword in ['acquisition', 'acquired', 'merger']):
            continue

        selected_stocks.append({
            "Ticker": ticker,
            "Company": info.get("shortName", ticker),
            "Sector": sector,
            "MarketCap": market_cap,
            "AvgVolume": avg_volume
        })

    except Exception as e:
        continue

# Step 4: 
df_selected = pd.DataFrame(selected_stocks)
df_selected = df_selected.sort_values("MarketCap", ascending=False)
n = len(df_selected)

df_selected["CapCategory"] = pd.cut(
    range(n),
    bins=[0, int(n * 0.1), int(n * 0.2), n],
    labels=["Top", "Mid", "Bottom"]
)

# Step 5: 
print(f"Targeted stocks：{len(df_selected)} ")
df_selected.reset_index(drop=True, inplace=True)
df_selected.style.background_gradient(subset=['MarketCap'], cmap='Blues')
ticker_list = df_selected["Ticker"].tolist()
print(ticker_list)

tickers = ['AAPL', 'AMZN', 'AVGO', 'ABBV', 'BAC', 'CVX', 'ABT', 'ACN', 'AXP', 'BX', 'AMD', 'ADBE', 'AMGN', 'BLK', 'BSX', 'SCHW', 'BMY',
           'AMAT', 'ANET', 'ADI', 'MO', 'AJG', 'AON', 'APH', 'APO', 'ABNB', 'CDNS', 'COF', 'BDX', 'AFL', 'BK', 'ADSK',
           'ALL', 'COR', 'AIG', 'BKR', 'ACGL', 'BRO', 'CAH', 'A', 'CNC', 'CCL', 'ADM',
           'CDW', 'BIIB', 'BAX', 'BBY', 'APTV', 'KMX', 'AKAM', 'ALGN', 'CPB', 'BG', 'TECH', 'APA', 'CRL', 'CZR']

financial_data = {}
for ticker in tickers:
    stock = yf.Ticker(ticker)
    financial_data[ticker] = {
        "EPS": stock.info.get("trailingEps", np.nan), 
        "ROE": stock.info.get("returnOnEquity", np.nan), 
        "Revenue Growth": stock.info.get("revenueGrowth", np.nan), 
        "Profit Growth": stock.info.get("grossMargins", np.nan), 
        "Book Value per Share": stock.info.get("bookValue", np.nan),
        "Beta": info.get("beta", np.nan),
        "Dividend Yield": info.get("dividendYield", np.nan)
    }

df = pd.DataFrame.from_dict(financial_data, orient='index')
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df_fin = df.copy()

# Step 2: standardize data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# Step 3: calculate adjacent matrix
distance_matrix = euclidean_distances(df_scaled.values) # calculate Eulindean distance for each pair of data
sigma = np.percentile(distance_matrix, 50)   # set sigma
W = rbf_kernel(df_scaled, gamma=1 / (2 * sigma ** 2)) # calculate the adjacent matrix: Wij = exp(-||xi - xj|| / 2* sigma ** 2)
n = W.shape[0]
D = np.diag(W.sum(axis=1)) # find the degree matrix

# Step 4: use three different spectral clustering methods
def spectral_unnormalized(W, D, k):
    L = D - W
    eigval, eigvec = eigsh(L, k=k, which='SM') # extract the first k smallest eigenvalues and eigenvectors
    return eigvec

def spectral_rw(W, D, k):
    L = D - W
    D_inv = np.linalg.inv(D)
    L_rw = D_inv @ L
    eigval, eigvec = eigsh(L_rw, k=k, which='SM')
    return eigvec

def spectral_sym(W, D, k):
    D_sqrt_inv = np.diag(1 / np.sqrt(np.diag(D)))
    L_sym = np.eye(n) - D_sqrt_inv @ W @ D_sqrt_inv
    eigval, eigvec = eigsh(L_sym, k=k, which='SM')
    Y = eigvec / np.linalg.norm(eigvec, axis=1, keepdims=True)
    return Y

# find the optimal k
def find_optimal_k(U, max_k=5):
    scores = []
    for k in range(2, max_k+1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(U)
        scores.append(silhouette_score(U, labels))
    optimal_k = np.argmax(scores) + 2 
    print(f"k values: {list(zip(range(2, max_k+1), scores))}, choose k={optimal_k}")
    return optimal_k

# use definitions to get eigenvectors
initial_k = 5
U_unn = spectral_unnormalized(W, D, initial_k)
U_rw = spectral_rw(W, D, initial_k)
U_sym = spectral_sym(W, D, initial_k)

optimal_k_unn = find_optimal_k(U_unn[:, :4])
optimal_k_rw = find_optimal_k(U_rw[:, :4])
optimal_k_sym = find_optimal_k(U_sym[:, :4])

k_opt = max(optimal_k_unn, optimal_k_rw, optimal_k_sym)

# Step 5: use k-means to do clustering
labels_unn = KMeans(n_clusters=k_opt, random_state=42).fit_predict(U_unn[:, :k_opt])
labels_rw = KMeans(n_clusters=k_opt, random_state=42).fit_predict(U_rw[:, :k_opt])
labels_sym = KMeans(n_clusters=k_opt, random_state=42).fit_predict(U_sym[:, :k_opt])

df['Cluster_Unnormalized'] = labels_unn
df['Cluster_RW'] = labels_rw
df['Cluster_Sym'] = labels_sym

# Step 6: visualization
def visualize_tsne_pca(U, labels, method_name):
    perplexity = min(5, U.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    pca = PCA(n_components=2)
    U_tsne = tsne.fit_transform(U)
    U_pca = pca.fit_transform(U)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"{method_name} - t-SNE")
    sns.scatterplot(x=U_tsne[:, 0], y=U_tsne[:, 1], hue=labels, palette='Set2', s=100)
    for i, txt in enumerate(df.index):
        plt.text(U_tsne[i, 0], U_tsne[i, 1], txt, fontsize=9)
    
    plt.subplot(1, 2, 2)
    plt.title(f"{method_name} - PCA")
    sns.scatterplot(x=U_pca[:, 0], y=U_pca[:, 1], hue=labels, palette='Set2', s=100)
    for i, txt in enumerate(df.index):
        plt.text(U_pca[i, 0], U_pca[i, 1], txt, fontsize=9)
    plt.tight_layout()
    plt.show()

visualize_tsne_pca(U_unn, labels_unn, "Unnormalized Spectral Clustering")
visualize_tsne_pca(U_rw, labels_rw, "RW Normalized Spectral Clustering")
visualize_tsne_pca(U_sym, labels_sym, "Symmetric Normalized Spectral Clustering")

summary_unn = df.groupby('Cluster_Unnormalized').mean()
summary_rw = df.groupby('Cluster_RW').mean()
summary_sym = df.groupby('Cluster_Sym').mean()

print("=== Unnormalized Spectral Clustering Summary ===")
print(summary_unn, '\n')
print("=== RW Normalized Spectral Clustering Summary ===")
print(summary_rw, '\n')
print("=== Symmetric Normalized Spectral Clustering Summary ===")
print(summary_sym, '\n')


df['Ticker'] = df.index
grouped = df.groupby('Cluster_Sym')

cluster_info = {}

for cluster_id, group in grouped:
    tickers = group.index.tolist()
    prices = []
    sectors = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("currentPrice", np.nan)
            sector = info.get("sector", "N/A")
        except:
            price = np.nan
            sector = "N/A"

        prices.append(price)
        sectors.append(sector)

    cluster_info[cluster_id] = pd.DataFrame({
        'Ticker': tickers,
        'Price': prices,
        'Sector': sectors,
        'Cluster': cluster_id
    })

full_table = pd.concat(cluster_info.values(), ignore_index=True)
display(full_table)




scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_fin), index=df_fin.index, columns=df_fin.columns)

# Compute the Euclidean distance matrix and then build an RBF kernel as adjacency
distance_matrix = euclidean_distances(df_scaled.values)
sigma = np.percentile(distance_matrix, 50)
W = rbf_kernel(df_scaled, gamma=1/(2*sigma**2))
n = W.shape[0]
D = np.diag(W.sum(axis=1))

def spectral_sym(W, D, k):
    D_sqrt_inv = np.diag(1 / np.sqrt(np.diag(D)))
    L_sym = np.eye(n) - D_sqrt_inv @ W @ D_sqrt_inv
    eigval, eigvec = eigsh(L_sym, k=k, which='SM')
    Y = eigvec / np.linalg.norm(eigvec, axis=1, keepdims=True)
    return Y

initial_k = 5
U_sym = spectral_sym(W, D, initial_k)

def find_optimal_k(U, max_k=5):
    scores = []
    for k in range(2, max_k+1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(U)
        score = silhouette_score(U, labels)
        scores.append(score)
    optimal_k = np.argmax(scores) + 2  # since k starts at 2
    print(f"k values: {list(zip(range(2, max_k+1), scores))}, choose k={optimal_k}")
    return optimal_k

optimal_k_sym = find_optimal_k(U_sym[:, :4])
labels_sym = KMeans(n_clusters=optimal_k_sym, random_state=42).fit_predict(U_sym[:, :optimal_k_sym])
df_fin['Cluster'] = labels_sym

print("Cluster counts:")
print(df_fin['Cluster'].value_counts())

# ----------------------------
# STEP 5: Cluster Summary for Risk Assessment
# ----------------------------
cluster_summary = df_fin.groupby('Cluster').agg({
    'Beta': 'mean',
    'Revenue Growth': 'mean',
    'Dividend Yield': 'mean',
    'EPS': 'mean',
    'ROE': 'mean'
}).reset_index()
print("Cluster Summary:")
print(cluster_summary)

# ----------------------------
# STEP 6: Assign Risk Profiles to Clusters
# ----------------------------
# We use average Beta as a simple risk proxy.
# Higher Beta suggests more volatility (thus Aggressive),
# Lower Beta (with higher dividend yield) suggests Conservative.
cluster_summary['RiskScore'] = cluster_summary['Beta']  # can be refined later

# Sort clusters by risk score (Beta)
cluster_summary = cluster_summary.sort_values('RiskScore', ascending=False).reset_index(drop=True)
print("Sorted Cluster Summary by Risk (Beta):")
print(cluster_summary)

# For simplicity, if we have at least 3 clusters:
if optimal_k_sym >= 3:
    aggressive_cluster = cluster_summary.loc[0, 'Cluster']
    conservative_cluster = cluster_summary.loc[optimal_k_sym - 1, 'Cluster']
    balanced_clusters = cluster_summary.loc[1:optimal_k_sym - 2, 'Cluster'].tolist()
else:
    # If only 2 clusters, assign one as aggressive and one as conservative.
    aggressive_cluster = cluster_summary.loc[0, 'Cluster']
    conservative_cluster = cluster_summary.loc[1, 'Cluster']
    balanced_clusters = []
    
print(f"Aggressive Cluster: {aggressive_cluster}")
print(f"Conservative Cluster: {conservative_cluster}")
print(f"Balanced Clusters: {balanced_clusters}")

# ----------------------------
# STEP 7: Build Portfolios Based on Investor Profiles
# ----------------------------
# We select top stocks based on:
# - Revenue Growth for aggressive (high growth in high-beta cluster)
# - A mix from balanced clusters (moderate growth)
# - Dividend Yield for conservative (to provide income and lower risk)

def select_top_stocks(df, cluster, metric, top_n=5, ascending=False):
    subset = df[df['Cluster'] == cluster]
    selected = subset.sort_values(metric, ascending=ascending).head(top_n)
    return selected

portfolio_aggressive = select_top_stocks(df_fin, aggressive_cluster, 'Revenue Growth', top_n=5, ascending=False)
portfolio_conservative = select_top_stocks(df_fin, conservative_cluster, 'Dividend Yield', top_n=5, ascending=False)

balanced_list = []
for cluster in balanced_clusters:
    balanced_list.append(select_top_stocks(df_fin, cluster, 'Revenue Growth', top_n=3, ascending=False))
if balanced_list:
    portfolio_balanced = pd.concat(balanced_list)
else:
    portfolio_balanced = pd.DataFrame()

# ----------------------------
# STEP 8: Enrich Recommendations with Current Market Info
# ----------------------------
def enrich_portfolio(df_portfolio):
    tickers = df_portfolio.index.tolist()
    prices = []
    sectors = []
    companies = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            prices.append(info.get("currentPrice", np.nan))
            sectors.append(info.get("sector", "N/A"))
            companies.append(info.get("shortName", ticker))
        except:
            prices.append(np.nan)
            sectors.append("N/A")
            companies.append(ticker)
    df_portfolio['Current Price'] = prices
    df_portfolio['Sector'] = sectors
    df_portfolio['Company'] = companies
    df_portfolio = df_portfolio.reset_index().rename(columns={'index': 'Ticker'})
    return df_portfolio

portfolio_aggressive = enrich_portfolio(portfolio_aggressive)
portfolio_balanced = enrich_portfolio(portfolio_balanced)
portfolio_conservative = enrich_portfolio(portfolio_conservative)

# ----------------------------
# Display the Recommended Portfolios
# ----------------------------
print("\n=== Aggressive Portfolio (College Age / Entry Level) ===")
print(portfolio_aggressive[['Ticker', 'Company', 'Current Price', 'Sector', 'Revenue Growth', 'Beta']])

print("\n=== Balanced Portfolio (Mid-Career) ===")
print(portfolio_balanced[['Ticker', 'Company', 'Current Price', 'Sector', 'Revenue Growth', 'Beta']])

print("\n=== Conservative Portfolio (Later Years) ===")
print(portfolio_conservative[['Ticker', 'Company', 'Current Price', 'Sector', 'Dividend Yield', 'Beta']])