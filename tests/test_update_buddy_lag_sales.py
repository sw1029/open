import numpy as np
import pandas as pd

from lstm_utils import update_buddy_lag_sales


def test_update_buddy_lag_sales_no_cross_sample():
    df = pd.DataFrame({
        'submission_date': ['TEST+1일', 'TEST+1일', 'TEST+2일', 'TEST+2일'],
        '영업장명_메뉴명': ['A', 'B', 'A', 'B'],
        'best_buddy': ['B', 'B', 'B', 'B'],
        'buddy_lag_1_sales': [np.nan, np.nan, np.nan, np.nan],
    })

    day_predictions = {'A': 10.0, 'B': 20.0}

    update_buddy_lag_sales(df, day_predictions, 'TEST+1일', {})

    a_mask = (df['submission_date'] == 'TEST+2일') & (df['영업장명_메뉴명'] == 'A')
    b_mask = (df['submission_date'] == 'TEST+2일') & (df['영업장명_메뉴명'] == 'B')

    assert np.isnan(df.loc[a_mask, 'buddy_lag_1_sales']).all()
    assert df.loc[b_mask, 'buddy_lag_1_sales'].iloc[0] == 20.0
