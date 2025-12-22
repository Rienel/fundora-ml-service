import pandas as pd
from datetime import datetime, timedelta
from .db_connection import DatabaseConnection

class DataCollector:
    def __init__(self):
        self.db = DatabaseConnection()
        self.db.connect()
    
    def collect_startups(self):
        """Collect all startup data"""
        query = """
        SELECT 
            s.id,
            s.company_name,
            s.industry,
            s.data_source_confidence,
            s.revenue,
            s.net_income,
            s.total_assets,
            s.total_liabilities,
            s.current_revenue,
            s.previous_revenue,
            s.confidence_percentage,
            s.is_deck_builder,
            s.current_valuation,
            s.expected_future_valuation,
            s.years_to_future_valuation,
            s.current_assets,
            s.current_liabilities,
            s.retained_earnings,
            s.ebit,
            s.created_at
        FROM core_startup s
        WHERE s.id IS NOT NULL
        """
        
        results = self.db.execute_query(query)
        df = pd.DataFrame(results)
        print(f"Collected {len(df)} startups")
        return df
    
    def collect_user_interactions(self, days=30):
        """Collect user viewing history"""
        query = """
        SELECT 
            sv.user_id,
            sv.startup_id,
            sv.viewed_at,
            CASE 
                WHEN w.id IS NOT NULL THEN 3
                WHEN sc.id IS NOT NULL THEN 2
                ELSE 1
            END as engagement_level
        FROM core_startupview sv
        LEFT JOIN core_watchlist w 
            ON sv.user_id = w.user_id 
            AND sv.startup_id = w.startup_id
        LEFT JOIN core_startupcomparison sc 
            ON sv.user_id = sc.user_id 
            AND sv.startup_id = sc.startup_id
        WHERE sv.viewed_at >= NOW() - INTERVAL '%s days'
        ORDER BY sv.viewed_at DESC
        """
        
        results = self.db.execute_query(query, (days,))
        df = pd.DataFrame(results)
        print(f"Collected {len(df)} user interactions from last {days} days")
        return df
    
    def collect_users(self):
        """Collect user profile data"""
        query = """
        SELECT 
            u.id,
            u.email,
            u.date_joined,
            ru.label
        FROM auth_user u
        LEFT JOIN core_registereduser ru ON u.id = ru.user_id
        """
        
        results = self.db.execute_query(query)
        df = pd.DataFrame(results)
        print(f"Collected {len(df)} users")
        return df
    
    def export_all_data(self):
        """Export all data to CSV files"""
        print("Starting data export...")
        
        # Collect data
        startups_df = self.collect_startups()
        interactions_df = self.collect_user_interactions()
        users_df = self.collect_users()
        
        # Export to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        startups_df.to_csv(f'data/raw/startups_{timestamp}.csv', index=False)
        interactions_df.to_csv(f'data/raw/interactions_{timestamp}.csv', index=False)
        users_df.to_csv(f'data/raw/users_{timestamp}.csv', index=False)
        
        print(f"✅ Data exported successfully to data/raw/")
        
        return {
            'startups': startups_df,
            'interactions': interactions_df,
            'users': users_df
        }
    
    def __del__(self):
        """Cleanup: close database connection"""
        self.db.close()

# Run this to test
if __name__ == "__main__":
    collector = DataCollector()
    data = collector.export_all_data()
    
    print("\nData Summary:")
    print(f"Startups: {len(data['startups'])} rows")
    print(f"Interactions: {len(data['interactions'])} rows")
    print(f"Users: {len(data['users'])} rows")