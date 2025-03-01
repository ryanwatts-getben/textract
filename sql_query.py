import pandas as pd
import os
import sys
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run SQL query and export results to CSV')
    parser.add_argument('--user', default='postgres', help='Database username')
    parser.add_argument('--password', default='postgres', help='Database password')
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', default=5432, type=int, help='Database port')
    parser.add_argument('--database', default='everycase', help='Database name')
    parser.add_argument('--table', default='ProjectAdditionalInfo', help='Table to query')
    parser.add_argument('--project-id', default='78585e4b-6234-4a28-9a70-c2ee6515a20e', help='Project ID to filter by')
    parser.add_argument('--output', default='query_results.csv', help='Output CSV file path')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Database connection parameters
    DB_PARAMS = {
        "user": args.user,
        "password": args.password,
        "host": args.host,
        "port": args.port,
        "database": args.database
    }
    
    try:
        # First attempt - direct psycopg2
        try:
            import psycopg2
            print("Attempting to connect using psycopg2...")
            
            conn = psycopg2.connect(**DB_PARAMS)
            print(f"Connected successfully to {args.database} using psycopg2")
            
            # Create engine for pandas
            from sqlalchemy import create_engine
            db_url = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"
            engine = create_engine(db_url)
            
        except ImportError:
            # If psycopg2 is not available, try SQLAlchemy directly
            print("psycopg2 not found, trying SQLAlchemy...")
            from sqlalchemy import create_engine
            
            # Create SQLAlchemy engine
            db_url = f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['database']}"
            engine = create_engine(db_url)
            conn = engine.connect()
            print(f"Connected successfully to {args.database} using SQLAlchemy")
        
        # Define your SQL query
        query = f'''
            SELECT *
            FROM "{args.table}"
            WHERE "projectId" = '{args.project_id}';
        '''
        
        # Read the query results into a pandas DataFrame
        print(f"Executing query on table {args.table}...")
        print(f"Query: {query}")
        
        # Use SQLAlchemy engine for pandas to avoid warnings
        df = pd.read_sql_query(query, engine)
        print(f"Query returned {len(df)} rows")
        
        if len(df) == 0:
            print("No results found. Please check if:")
            print(f"1. The table '{args.table}' exists")
            print(f"2. The projectId '{args.project_id}' exists in the table")
            
            # List available tables for troubleshooting
            try:
                inspector_query = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
                tables_df = pd.read_sql_query(inspector_query, engine)
                print("\nAvailable tables in database:")
                for table in tables_df['table_name'].tolist():
                    print(f"- {table}")
            except Exception as table_error:
                print(f"Could not list tables: {str(table_error)}")
        
        # Write the DataFrame to a CSV file without the index column
        df.to_csv(args.output, index=False, encoding='utf-8')
        
        # Close the connection
        if hasattr(conn, 'close'):
            conn.close()
        if hasattr(engine, 'dispose'):
            engine.dispose()
        
        print(f"Query results have been written to '{args.output}' using pandas")
        print(f"File saved at: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
        # Provide troubleshooting information
        print("\nTroubleshooting suggestions:")
        print("1. Verify that your PostgreSQL server is running")
        print(f"2. Check that the database '{args.database}' exists")
        print("3. Verify your username and password")
        print(f"4. Make sure the host and port are correct ({args.host}:{args.port})")
        
        # List available packages for debugging
        print("\nInstalled database packages:")
        try:
            import pkg_resources
            for pkg in ['psycopg2', 'psycopg2-binary', 'sqlalchemy', 'pg8000', 'pymysql']:
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    print(f"- {pkg}: {version}")
                except pkg_resources.DistributionNotFound:
                    pass
        except ImportError:
            print("Could not check installed packages")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
