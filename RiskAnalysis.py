import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_stock_data(tickers, period='2y'):
    """
    Belirtilen hisselerin geçmiş verilerini yfinance'ten çeker.
    """
    print("Veriler çekiliyor...")
    # 'Close' sütununu kullanarak veriyi çek ve 'dropna' ile eksik verileri temizle
    data = yf.download(tickers, period=period)['Close'].dropna()
    print("Veri çekme tamamlandı.")
    return data

def calculate_statistics(log_returns):
    """
    Logaritmik getirilerden istatistiksel metrikleri hesaplar.
    (Yıllık Kovaryans Matrisi, Günlük Drift, Günlük Cholesky Matrisi)
    """
    # Yıllık bazda Kovaryans Matrisi (252 işlem günü)
    cov_matrix_annual = log_returns.cov() * 252
    
    # Her hissenin ortalama günlük getirisi (mu)
    mu_vector = log_returns.mean().values
    
    # Cholesky ayrıştırması için 'L' matrisi (yıllık)
    L_annual = np.linalg.cholesky(cov_matrix_annual)
    
    # Simülasyon için günlük parametreler
    daily_drift = mu_vector - 0.5 * np.diag(cov_matrix_annual / 252)
    daily_L = L_annual / np.sqrt(252)
    
    return daily_drift, daily_L, cov_matrix_annual

def run_portfolio_simulation(tickers, weights, S0_vector, P0, T, M, daily_drift, daily_L):
    """
    Monte Carlo simülasyonunu korelasyonlu olarak çalıştırır.
    """
    print(f"{M} senaryo için simülasyon başlatılıyor...")
    num_assets = len(tickers)
    
    # Portföy değerlerini saklamak için (Gün, Senaryo)
    portfolio_values = np.zeros((T, M))
    portfolio_values[0] = P0
    
    # Hisse fiyat yollarını saklamak için (Gün, Senaryo, Hisse Sayısı)
    price_paths = np.zeros((T, M, num_assets))
    price_paths[0, :, :] = S0_vector
    
    for t in range(1, T):
        # Korelasyonsuz rastgele sayılar (M senaryo, num_assets hisse)
        Z = np.random.standard_normal((M, num_assets))
        
        # Korelasyonlu rastgele sayılar
        correlated_Z = Z @ daily_L.T
        
        # Her hissenin yeni fiyatlarını hesapla
        price_paths[t] = price_paths[t-1] * np.exp(daily_drift + correlated_Z)
        
        # O günkü portföyün toplam değerini hesapla
        # (Fiyatlar @ Ağırlıklar) / (Başlangıç Fiyatları @ Ağırlıklar) * Başlangıç Portföy Değeri
        # Bu, portföyün yeniden dengelenmediğini varsayar, sadece değeri izler.
        current_portfolio_value = (price_paths[t] @ weights) / (price_paths[0] @ weights) * P0
        portfolio_values[t] = current_portfolio_value

    print("Simülasyon tamamlandı.")
    return portfolio_values, price_paths

def calculate_risk_metrics(portfolio_values, P0):
    """
    Simülasyon sonuçlarından VaR ve CVaR metriklerini hesaplar.
    """
    # Sadece vade sonu (T. gün) değerlerine odaklan
    final_portfolio_values = portfolio_values[-1]
    
    # Başlangıç değerine göre Kâr/Zarar
    returns = final_portfolio_values - P0
    
    # %95 Güvenle VaR (Value-at-Risk)
    VaR_95 = np.percentile(returns, 5)
    
    # %95 Güvenle CVaR (Conditional VaR)
    CVaR_95 = returns[returns <= VaR_95].mean()
    
    metrics = {
        'initial_value': P0,
        'mean_final_value': final_portfolio_values.mean(),
        'min_scenario': final_portfolio_values.min(),
        'max_scenario': final_portfolio_values.max(),
        'VaR_95': VaR_95,
        'CVaR_95': CVaR_95
    }
    return metrics, final_portfolio_values, returns

def plot_simulation_results(portfolio_values, final_portfolio_values, returns, metrics, tickers_str, T, M):
    """
    Simülasyon sonuçlarını ve risk metriklerini görselleştirir.
    Grafikleri 'portfolio_paths.png' ve 'portfolio_histogram.png' olarak kaydeder.
    """
    print("Grafikler oluşturuluyor ve kaydediliyor...")
    
    P0 = metrics['initial_value']
    VaR_95 = metrics['VaR_95']
    CVaR_95 = metrics['CVaR_95']
    
    # 1. Grafik: Tüm Olası Portföy Değer Yolları
    plt.figure(figsize=(12, 7))
    plt.plot(portfolio_values)
    plt.title(f'{tickers_str} Portföyü için {M} Simülasyon ({T} Gün)')
    plt.xlabel('İşlem Günü')
    plt.ylabel('Portföy Değeri (TL)')
    plt.grid(True)
    plt.savefig('portfolio_paths.png') # Grafiği kaydet
    plt.close() # Grafiği kapat (gösterme)

    # 2. Grafik: Vade Sonu Portföy Değer Dağılımı (Histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(final_portfolio_values, bins=100, edgecolor='black', alpha=0.7, label='Portföy Değer Dağılımı')
    
    plt.title(f'{T}. Gün Sonundaki Portföy Değer Dağılımı')
    plt.xlabel('Değer (TL)')
    plt.ylabel('Frekans (Senaryo Sayısı)')
    
    # VaR ve CVaR çizgileri
    VaR_line_value = P0 + VaR_95
    CVaR_line_value = P0 + CVaR_95

    plt.axvline(VaR_line_value, color='red', linestyle='dashed', linewidth=2, 
                label=f'%95 VaR Değeri: {VaR_line_value:,.2f} TL')
    plt.axvline(CVaR_line_value, color='purple', linestyle='dashed', linewidth=2, 
                label=f'%95 CVaR Değeri: {CVaR_line_value:,.2f} TL')
    plt.axvline(P0, color='black', linestyle='solid', linewidth=2, label=f'Başlangıç Değeri: {P0:,.2f} TL')
    plt.legend()
    plt.savefig('portfolio_histogram.png') # Grafiği kaydet
    plt.close() # Grafiği kapat
    
    print("Grafikler başarıyla 'portfolio_paths.png' ve 'portfolio_histogram.png' olarak kaydedildi.")

def main():
    """
    Ana program akışı.
    """
    # --- Parametreler ---
    tickers = ['THYAO.IS', 'ASELS.IS']
    weights = np.array([0.60, 0.40])
    tickers_str = f'%{weights[0]*100} {tickers[0]}, %{weights[1]*100} {tickers[1]}'
    
    P0 = 100_000  # Başlangıç portföy değeri (100.000 TL)
    T = 252       # Simülasyon süresi (1 yıl)
    M = 5000      # Senaryo sayısı
    
    # 1. Veri Çekme
    data = get_stock_data(tickers, period='2y')
    
    # 2. İstatistikleri Hesaplama
    log_returns = np.log(1 + data.pct_change()).dropna()
    daily_drift, daily_L, cov_matrix = calculate_statistics(log_returns)
    
    print("Yıllık Kovaryans Matrisi:\n", cov_matrix)
    
    # 3. Simülasyonu Çalıştırma
    S0_vector = data.iloc[-1].values # Başlangıç fiyatları
    portfolio_values, price_paths = run_portfolio_simulation(
        tickers, weights, S0_vector, P0, T, M, daily_drift, daily_L
    )
    
    # 4. Risk Metriklerini Hesaplama
    metrics, final_values, returns = calculate_risk_metrics(portfolio_values, P0)
    
    # 5. Sonuçları Raporlama (Terminal)
    print("\n--- RİSK ANALİZİ (1 Yıl Sonrası) ---")
    print(f"Portföy: {tickers_str}")
    print(f"Başlangıç Portföy Değeri: {metrics['initial_value']:,.2f} TL")
    print("-" * 35)
    print(f"Simülasyon Sonu Ortalama Değer: {metrics['mean_final_value']:,.2f} TL")
    print(f"En Kötü Senaryo (Min): {metrics['min_scenario']:,.2f} TL")
    print(f"En İyi Senaryo (Max): {metrics['max_scenario']:,.2f} TL")
    print("-" * 35)
    print(f"%95 VaR: {metrics['VaR_95']:,.2f} TL")
    print(f"   -> Anlamı: %95 ihtimalle, kaybınız bu tutardan *az* olacaktır.")
    print(f"\n%95 CVaR: {metrics['CVaR_95']:,.2f} TL")
    print(f"   -> Anlamı: İşler kötü giderse (%5'lik dilim), beklenen *ortalama* kaybınız bu kadar olacaktır.")
    
    # 6. Sonuçları Görselleştirme (Dosyaya Kaydetme)
    plot_simulation_results(portfolio_values, final_values, returns, metrics, tickers_str, T, M)

# Bu, script'in doğrudan çalıştırıldığında 'main' fonksiyonunu çağırmasını sağlar.
if __name__ == "__main__":
    main()