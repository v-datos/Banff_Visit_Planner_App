import requests
import zipfile
import io
import pantab
import pandas as pd
import tempfile
import os

def main():
    url = "https://public.tableau.com/workbooks/BanffTrafficData-GS.twb"
    r = requests.get(url)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        hyper_files = [name for name in z.namelist() if name.endswith(".hyper")]

        if len(hyper_files) != 1:
            raise ValueError(
                f"Expected exactly one .hyper extract, found {len(hyper_files)}: {hyper_files}"
            )

        name = hyper_files[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".hyper") as temp_file:
            temp_file.write(z.read(name))
            temp_path = temp_file.name

        try:
            frames = pantab.frames_from_hyper(temp_path)

            dfs = {}
            for table_name, df in frames.items():
                if "Counter" in df.columns:
                    c_val = str(df["Counter"].iloc[0]).strip()
                    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
                    df['Date'] = df['TimeStamp'].dt.date
                    daily = df.groupby('Date')['TW'].apply(lambda x: pd.to_numeric(x, errors='coerce').sum()).reset_index()
                    dfs[c_val] = daily

            # Merge data
            # Required columns: Day of Time Stamp, Weekday, Combined TW, Tw (002), Tw (003), Sum of TW
            # Where Combined TW = Counter 1 + Counter 2, Sum of TW = Counter 1

            merged = dfs.get('1', pd.DataFrame(columns=['Date', 'TW'])).rename(columns={'TW': 'Sum of TW'})

            if '2' in dfs:
                merged = pd.merge(merged, dfs['2'].rename(columns={'TW': 'Tw (002)'}), on='Date', how='outer')
            else:
                merged['Tw (002)'] = 0

            if '3' in dfs:
                merged = pd.merge(merged, dfs['3'].rename(columns={'TW': 'Tw (003)'}), on='Date', how='outer')
            else:
                merged['Tw (003)'] = 0

            merged.fillna(0, inplace=True)

            # Convert columns to int
            for col in ['Sum of TW', 'Tw (002)', 'Tw (003)']:
                merged[col] = merged[col].astype(int)

            merged['Combined TW'] = merged['Tw (002)'] + merged['Sum of TW']

            # Filter dates from July 2013 onwards (matching original data)
            import datetime
            start_date = datetime.date(2013, 7, 1)
            merged = merged[merged['Date'] >= start_date].copy()

            merged.sort_values('Date', inplace=True)

            merged['Day of Time Stamp'] = pd.to_datetime(merged['Date']).dt.strftime('%B %-d, %Y')
            merged['Weekday'] = pd.to_datetime(merged['Date']).dt.strftime('%a')

            final_df = merged[['Day of Time Stamp', 'Weekday', 'Combined TW', 'Tw (002)', 'Tw (003)', 'Sum of TW']]

            dummy_row = pd.DataFrame([{
                'Day of Time Stamp': 'Grand Total',
                'Weekday': '',
                'Combined TW': final_df['Combined TW'].sum(),
                'Tw (002)': final_df['Tw (002)'].sum(),
                'Tw (003)': final_df['Tw (003)'].sum(),
                'Sum of TW': final_df['Sum of TW'].sum()
            }])
            final_df = pd.concat([final_df, dummy_row], ignore_index=True)

            # Save as UTF-16 TSV
            # To match original format closely
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TW Traffic _data.csv')
            final_df.to_csv(output_path, index=False, sep='\t', encoding='utf-16')
            print(f"Data successfully extracted and saved to {output_path}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    main()
