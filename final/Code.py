import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# STEP 1: Get S&P 500 Stocks
# ----------------------------
wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(wiki_url)
sp500_df = tables[0]
tickers = sp500_df['Symbol'].tolist()
industries = sp500_df.set_index('Symbol')['GICS Sector'].to_dict()

# ----------------------------
# STEP 2: Filter Stocks
# ----------------------------
target_sectors = [
    'Information Technology', 'Consumer Discretionary', 'Consumer Staples',
    'Financials', 'Health Care', 'Energy'
]
liquidity_threshold = 1_000_000
max_companies_to_check = 200  
selected_stocks = []

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
        # Exclude names with keywords that might indicate M&A situations
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

df_selected = pd.DataFrame(selected_stocks)
df_selected = df_selected.sort_values("MarketCap", ascending=False)
ticker_list = df_selected["Ticker"].tolist()
print(f"Targeted stocks: {len(ticker_list)} stocks")
print(ticker_list)

# ----------------------------
# STEP 3: Download Financial Data
# ----------------------------
# We include Beta and Dividend Yield along with other metrics.
financial_data = {}
for ticker in ticker_list:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financial_data[ticker] = {
            "EPS": info.get("trailingEps", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "Revenue Growth": info.get("revenueGrowth", np.nan),
            "Profit Growth": info.get("grossMargins", np.nan),
            "Book Value per Share": info.get("bookValue", np.nan),
            "Beta": info.get("beta", np.nan),
            "Dividend Yield": info.get("dividendYield", np.nan)
        }
    except Exception as e:
        continue

df_fin = pd.DataFrame.from_dict(financial_data, orient='index')
df_fin = df_fin.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Financial data available for {len(df_fin)} stocks")

# ----------------------------
# STEP 4: Clustering via Spectral Clustering (Symmetric)
# ----------------------------
# Standardize the features
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

# ----------------------------
# OPTIONAL: Visualize the Clusters using t-SNE or PCA
# ----------------------------
# For example, using PCA:
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
U_pca = pca.fit_transform(df_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=U_pca[:, 0], y=U_pca[:, 1], hue=df_fin['Cluster'], palette='Set2', s=100)
plt.title("PCA of Standardized Financial Data Colored by Cluster")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()
