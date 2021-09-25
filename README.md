# backtest_skeleton

FR Coding Backtest Module Skeleton Code + Financial Data CSV Files Database

-Financial Data CSV Files-
  1. AAPL.csv (sample data file)
  
    기간: 20070103 - 20200722
    주기: 일별 데이터
    포함 내역: Date, Open, High, Low, Close, Volume, Net Profit Margin, Sales Growth QOQ, Price to Book Value
  
  2. 21개 data 
    
    파일명: 
      (1) Adj. Close_data.csv
      (2) Assets Growth_data.csv
      (3) Close_data.csv
      (4) Debt Ratio_data.csv
      (5) Dividend Yield_data.csv
      (6) Earnings Growth_data.csv
      (7) Earnings Yield_data.csv
      (8) Gross Profit Margin_data.csv
      (9) High_data.csv
      (10) Low_data.csv
      (11) Market-Cap_data.csv
      (12) Net Profit Margin_data.csv
      (13) Open_data.csv
      (14) PDIVCash_data.csv
      (15) PDIVE_data.csv
      (16) PDIVSales_data.csv
      (17) Price to Book Value_data.csv
      (18) Return on Assets_data.csv
      (19) Return on Equity_data.csv
      (20) Sales Growth_data.csv
      (21) Volume_data.csv
      
    기간: 20131001 - 20200914 (총 1751일)
    주기: 일별 데이터
    대상: 미국 주식 446개 종목
    구성: 각 파일별로 행(일), 열(종목)
    
    -> DIV는 '/'를 의미
    (예: PDIVE = P/E)
    -> 밑의 링크를 통해 각 데이터가 무엇을 의미하는지 상세확인 가능
    https://github.com/SimFin/simfin-tutorials/blob/master/04_Signals.ipynb
    
   
-Python Code-

  backtest_module.py
  : 백테스팅 모듈 스켈레톤 소스코드 파일
    
