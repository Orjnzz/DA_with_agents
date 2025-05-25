import pandas as pd

def info_dataframe(dataframe):
        # string_data = '\n'.join([' | '.join(map(str, row)) for row in dataframe.values])
        sample_df = dataframe.sample(n=min(5, len(dataframe)))
        header = ' | '.join(sample_df.columns)
        rows = [' | '.join(map(str, row)) for row in sample_df.values]
        sample_data = header + '\n' + '\n'.join(rows)
        data_desc = {
            # "columns": list(dataframe.columns),
            "sample dataframe": sample_data,
            "stats": dataframe.describe().to_dict(),
        }
        return data_desc

def preprocess_df(
    df: pd.DataFrame,
    strip_columns: bool = True,
    snake_case_columns: bool = True,
    collapse_spaces: bool = True,
    unify_missing: bool = True,
    missing_values: list = ['', 'NA', 'NaN', None],
    fill_numeric_strategy: str = 'median',  # 'mean'/'median'
    fill_categorical_value: str = 'Unknown',
    drop_columns: list = None,
    rename_columns: dict = None,
    generate_id: bool = True,
    id_column: str = 'id'
) -> pd.DataFrame:
    """
    Hàm preprocess chung cho bất kỳ DataFrame nào:
    - (1) Chuẩn hóa tên cột: strip, snake_case
    - (2) Trim và collapse spaces cho string
    - (3) Thống nhất giá trị missing
    - (4) Fill missing numeric/categorical
    - (5) Drop hoặc rename cột tuỳ chọn
    - (6) Tạo cột id nếu cần
    Tham số:
    - df: DataFrame đầu vào
    - strip_columns: loại bỏ whitespace khỏi tên cột
    - snake_case_columns: chuyển tên cột thành snake_case
    - collapse_spaces: gộp nhiều khoảng trắng thành 1
    - unify_missing: chuyển giá trị trong missing_values thành pd.NA
    - missing_values: danh sách giá trị xem như missing
    - fill_numeric_strategy: cách fill số ('mean' hoặc 'median')
    - fill_categorical_value: giá trị fill cho categorical
    - drop_columns: danh sách cột muốn drop
    - rename_columns: dict {old_name: new_name}
    - generate_id: True để sinh cột id
    - id_column: tên cột id sinh ra
    Trả về DataFrame đã xử lý.
    """
    df = df.copy()
    
    # 1. Normalize column names
    if strip_columns:
        df.columns = df.columns.str.strip()
    if snake_case_columns:
        df.columns = (
            df.columns
              .str.lower()
              .str.replace(r'\s+', '_', regex=True)
              .str.replace(r'[^\w_]', '', regex=True)
        )
    
    # 2. Xử lý text columns
    if collapse_spaces:
        for col in df.select_dtypes(include='object').columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )
    
    # 3. Thống nhất missing
    if unify_missing:
        df.replace(missing_values, pd.NA, inplace=True)
    
    # 4. Fill missing
    # Numeric
    num_cols = df.select_dtypes(include='number').columns
    for col in num_cols:
        if fill_numeric_strategy == 'mean':
            val = df[col].mean()
        else:
            val = df[col].median()
        df[col] = df[col].fillna(val)
    
    # Categorical
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(fill_categorical_value)
    
    # 5. Drop & rename columns
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
    if rename_columns:
        df.rename(columns=rename_columns, inplace=True)
    
    # 6.Generate id column
    if generate_id:
        df[id_column] = range(1, len(df) + 1)
    
    return df


