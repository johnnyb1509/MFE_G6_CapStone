# This Python file contains functions for working with SQL Server
import numpy as np
import pandas as pd
import logging
# CONNECT TO DB
class dbJob():
    # TODO: dev DELETE and REPLACE function for this class
    def __init__(self, server, database, username, password, driver):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.driver = driver
        
        self.engine, self.tableInDB = self.sqlServerConnect()
        
    def sqlServerConnect(self):
        """Function for creating engine to connect to the SQL
            AND get all table name which included in the given database
            Using SQLAlchemy
        Args:
            database (str): database target name
            username (str): user account name
            password (str): password
            driver (str): driver name 

        Returns:
            engine (sqlalchemy object): engine for connecting
        """
        # import pyodbc
        from sqlalchemy import create_engine, inspect
        from sqlalchemy.engine import URL
        # conn = pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
        connection_string = f"DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
        connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
        engine = create_engine(connection_url, pool_pre_ping=True, pool_size = -1)

        list_tableInDB = inspect(engine).get_table_names()
        return engine, list_tableInDB
    
    def engineDispose(self):
        return self.engine.dispose()
    
    def set_primary_key(self, targetTableName):
        logging.info(f"SET PRIMARY KEY FOR TABLE {targetTableName}")
        change_notnull = f"ALTER TABLE dbo.{targetTableName} ALTER COLUMN id_date INTEGER NOT NULL;"
        add_primaryKey = f"ALTER TABLE dbo.{targetTableName} ADD PRIMARY KEY (id_date);"
        with self.engine.connect() as conn:
            conn.execute(change_notnull)
            conn.execute(add_primaryKey)
            logging.info(f"SET PRIMARY KEY FOR TABLE {targetTableName}: SUCCESSFUL!")
            conn.close()
        return

    def getData(self, tableName, columns=None):
        """Read table from SQL Server at self.database

        Args:
            tableName (str): Name of the table that wanted to get
            connection (SQLAlchemy object): self.engine.connect()

        Returns:
            pd.DataFrame
        """
        with self.engine.connect() as conn:
            table = pd.read_sql_table(tableName, conn, columns=columns)
            conn.close()
        return table
    
    def getData_slice(self, arg_choice, tableName, is_exact = False):
        """Get little data just for checking columns name and <n_row> latest data
            NOTICE: JUST GET TABLE WITH id_date AS PRIMARY KEY
        

        Args:
            arg_choice (int): If is_exact == False -> Number of rows want to get 
                                        (e.g 10 rows or 100 rows)
                            Elif is_exact == True -> id_date number of row want to get 
                                        (e.g row with id_date == <arg_choice>)
            tableName (str): Name of the table
        """
        short_table = None
        if is_exact == False:
            getData_query = f"SELECT TOP({arg_choice}) * FROM {self.database}.dbo.{tableName} ORDER BY id_date DESC;"
            short_table = pd.read_sql(getData_query, self.engine)
        elif is_exact == True:
            getExact_query = f"SELECT * FROM {self.database}.dbo.{tableName} WHERE id_date = {arg_choice};"
            short_table = pd.read_sql(getExact_query, self.engine)
        return short_table

    def insertData(self, dataTable, targetTableName, dtypes = None):
        """Insert data from pd.DataFrame to table on SQL Server using SQLALchemy

        Args:
            dataTable (pd.DataFrame): DateFrame that includes row (rows) to insert.
                                        The order of columns is not important
            targetTableName (str): name of the table that included in self.database
        """
        check_table = self.checkTableExist(targetTableName) # Check if targetTableName exist
        if check_table == True:
            logging.info("---------------------------------------")
            logging.info(f"INSERTING DATA INTO {targetTableName}")
            logging.info("CHECKING COLUMNS DIFFERENCE")
            
            # Pull before push to compare columns
            df_cache = self.getData_slice(1000, targetTableName, is_exact=False)
            list_old_columns = list(df_cache.columns)
            list_new_columns = list(dataTable.columns)
            list_diff = [item for item in list_new_columns if item not in list_old_columns]# list(set(list_new_columns) - set(list_old_columns)) # component in list_new but not
                                                                            # in list_old
            list_diff_type = list()

            if len(list_diff) != 0: # if there is any new columns, create T-SQL type for columns
                logging.info("APPEAR COLUMNS DIFFERENCE, ADDING COLUMN(s)")
                for new_col in list_diff:
                    type_ = dataTable[new_col].dtypes
                    if type_ == "float64":
                        list_diff_type.append("FLOAT")
                    elif type_ == "int64":
                        list_diff_type.append("INT")
                    elif type_ == "O":
                        list_diff_type.append("DATE")
                    else:
                        list_diff_type.append("VARCHAR")
                        
                for new_col_idx in range(len(list_diff)):
                    logging.info(f"--------- ADDING COLUMN(s) {list_diff[new_col_idx]}")
                    self.addColumn(targetTableName, list_diff[new_col_idx], list_diff_type[new_col_idx])
                
            is_overlap_primaryKey = self.checkPrimaryKeyExist(dataTable['id_date'].values[0], df_cache)
            if is_overlap_primaryKey == False:
                logging.info("NO OVERLAP OF PRIMARY KEY, INSERTING DATA INTO TABLE")
                dataTable.to_sql(targetTableName, self.engine.connect(), index=False, if_exists = 'append',
                                    dtype = dtypes, chunksize=20000)
            elif is_overlap_primaryKey == True:
                logging.info("FOUND OVERLAP OF PRIMARY KEY")
                is_diffData, self.data_checkDict = self.compareDataRow(dataTable, targetTableName)
                # update if has new columns
                if len(list_diff) != 0: 
                    logging.info("APPEAR NEW COLUMN(S), INSERTING NEW COLUMN(S)")
                    for new_col in list_diff:
                        self.updateData(targetTableName, dataTable['id_date'].values[0], 
                                        dataTable, dataTable[new_col].values[0])
                elif len(list_diff) == 0:
                    logging.info(f"NO NEW COLUMN TO PUSH")
                    
                # update if has different data
                if is_diffData == True:
                    logging.info("FOUND DIFFERENT IN DATA, UPDATING COLUMNS DATA")
                    for column, value in self.data_checkDict.items():
                        logging.info(f"------ UPDATING COLUMNS {column}")
                        self.updateData(targetTableName, dataTable['id_date'].values[0], 
                                        column, value)
                else:
                    logging.info(f"ALL DATA VALUE IS THE SAME FOR id_date {dataTable['id_date'].values[0]}")
                   
            logging.info(f"INSERT DATA INTO {targetTableName} FINISH")
            logging.info("---------------------------------------")
            
        elif check_table == False: # if not > creating new table
            logging.info(f"TABLE {targetTableName} DOES NOT EXIST IN DATABASE {self.database}")
            logging.info(f"CREATING NEW TABLE {targetTableName}")
            self.createNewTable(dataTable, targetTableName)
        return
    
    
    def insertData_VBMADanhMuc(self, dataTable, targetTableName, dtypes = None):
        """Insert data from pd.DataFrame to table 'debt_danhmuc_rawData' - 'debt_data'
                on SQL Server using SQLALchemy

        Args:
            dataTable (pd.DataFrame): DateFrame that includes row (rows) to insert.
                                        The order of columns is not important
            targetTableName (str): name of the table that included in self.database
        """
        check_table = self.checkTableExist(targetTableName) # Check if targetTableName exist
        if check_table == True:
            logging.info("---------------------------------------")
            logging.info(f"INSERT DATA INTO {targetTableName}")
            logging.info("CHECKING COLUMNS DIFFERENCE")
            
            # Pull before push to compare columns
            df_cache = self.getData(targetTableName)
            list_old_columns = list(df_cache.columns)
            list_new_columns = list(dataTable.columns)
            list_diff = [item for item in list_new_columns if item not in list_old_columns]# list(set(list_new_columns) - set(list_old_columns)) # component in list_new but not
                                                                            # in list_old
            list_diff_type = list()

            if len(list_diff) != 0: # if there is any new columns, create T-SQL type for columns
                logging.info("APPEAR COLUMNS DIFFERENCE, ADDING COLUMNS")
                for new_col in list_diff:
                    type_ = dataTable[new_col].dtypes
                    if type_ == "float64":
                        list_diff_type.append("FLOAT")
                    elif type_ == "int64":
                        list_diff_type.append("INT")
                    elif type_ == "O":
                        list_diff_type.append("DATE")
                    else:
                        list_diff_type.append("VARCHAR")
                        
                for new_col_idx in range(len(list_diff)):
                    logging.info(f"--------- ADDING COLUMN(s) {list_diff[new_col_idx]}")
                    self.addColumn(targetTableName, list_diff[new_col_idx], list_diff_type[new_col_idx])
                    
            check_unique_db = list(df_cache['ngay_TCPH'].unique())
            check_unique_newData = list(dataTable['ngay_TCPH'].unique())
            
            delete_row = list()
            logging.info("CHECKING UNIQUE ngay_TCPH")
            for new in check_unique_newData:
                if new in check_unique_db:
                    delete_row.append(new)
            for del_row in delete_row:
                dataTable = dataTable.drop(dataTable.index[dataTable['ngay_TCPH'] == del_row])
                
            logging.info(f"DATA PUSH \n {dataTable}")
            if len(dataTable) > 0:
                logging.info(f"THERE ARE {len(dataTable)} NEW ROW(S) TO PUSH")
                dataTable.to_sql(targetTableName, self.engine.connect(), index=False, if_exists = 'append',
                                        dtype = dtypes, chunksize=20000)
                logging.info(f"INSERT DATA INTO {targetTableName} SUCCESSFUL")
                logging.info("---------------------------------------")
            else:
                logging.info("NO NEW DATA TO PUSH")
            
        elif check_table == False: # if not > creating new table
            logging.info(f"TABLE {targetTableName} DOES NOT EXIST IN DATABASE {self.database}")
            logging.info(f"CREATING NEW TABLE {targetTableName}")
            self.createNewTable(dataTable, targetTableName)
        return
    
    def addColumn(self, targetTableName, columnName, data_type):
        """Add new columns and define its datatype 

        Args:
            targetTableName (str): Name of table
            columnName (_type_): _description_
            data_type (_type_): _description_
        """
        query = f"ALTER TABLE [dbo].[{targetTableName}] ADD [{columnName}] {data_type} NULL"
        with self.engine.connect() as conn:
            conn.execute(query)
            conn.close()
        return
    
    def deleteData(self, table, condition):
        """Delete row with the defined condition on 'id_date' column

        Args:
            table (string): table that need to be affected 
            condition (string): e.g "= 20130405"
        """
        logging.info(f"DELETE ROW id_date = {condition} IN TABLE {table}")
        query = f"DELETE FROM [dbo].[{table}] WHERE id_date {condition};"
        with self.engine.connect() as conn:
            conn.execute(query)
            conn.close()
        return
    
    def createNewTable(self, dataTable, targetTableName, dtypes = None):
        """Create new table on SQL Server then add Primary Key to the targetTableName

        Args:
            dataTable (pd.DataFrame): new table that want to push
            targetTableName (String): Name of the table
            dtypes (dict, optional): Specify DataType of each columns. Defaults to None.
        """
        logging.info(f"CREATE/REPLACE TABLE DATA: {targetTableName}")
        # Push table to sql
        if dtypes == None: # Auto create data type using SQLAlchemy datatype
            import sqlalchemy
            dtypes = dict()
            for col in dataTable.columns:
                if col == "id_date":
                    dtypes[col] = sqlalchemy.types.Integer()
                elif col == "date":
                    dtypes[col] = sqlalchemy.types.DATE()
                else:
                    dtypes[col] = sqlalchemy.types.Float()
        
        dataTable.to_sql(targetTableName, self.engine.connect(), index = False,
                            if_exists = 'replace', dtype = dtypes, chunksize=20000)

        self.set_primary_key(targetTableName)
            
        logging.info(f"CREATE/REPLACE TABLE DATA {targetTableName} SUCCESSFUL")
        logging.info("---------------------------------------")
        return
    
    def updateData(self, targetTableName, id_date_cond, update_column, update_value):
        """Update data for specific row (depends on id_date value) for ex

        Args:
            targetTableName (str): name of the table in self.database.
            id_date_cond (int): value of id_date column (PRIMARY KEY,)
            update_column (str): name of the columns that want to update
            update_value (float): value to be updated
        """
        logging.info("UPDATING DATA")
        # logging.info(f"+_+_+_+_+_ {update_value}+_+_+_+_+")
        # logging.info(f"+_+_+_+_+_ {type(update_value)}+_+_+_+_+")
        assert type(update_column) == str
        update_query = f"UPDATE [dbo].[{targetTableName}] \
                        SET [{update_column}] = {update_value} \
                        WHERE id_date = {id_date_cond}"
        
        if update_value is None or np.isnan(update_value): # cannot set the value = nan directly, SQL Server raise Error
            logging.info("CHANGE NAN VALUE TO COMPATIBLE VALUE")
            update_query = f"UPDATE [dbo].[{targetTableName}] \
                            SET [{update_column}] = NULL \
                            WHERE id_date = {id_date_cond}"
                            
        with self.engine.connect() as conn:
            conn.execute(update_query)
            conn.close()
        return
    
    def compareDataRow(self, newTable, targetTableName):
        """This function tend to check data each column for only one specific row.
            USE CASE: check if there are any duplicate value between old and new data

        Args:
            newTable (pd.DataFrame): table of new data
            targetTableName (str): name of the table on database

        Returns:
            is_diff (bool): True if there are any different value, else False
            data_cache (dict): dictionary of data.
        """
        logging.info("COMPARING OLD DATA WITH NEW DATA")
        oldTable = self.getData_slice(newTable['id_date'].values[0], targetTableName, is_exact=True)
        # logging.info(f"DEBUGGING ---- OLD TABLE {oldTable}")
        # logging.info(f"DEBUGGING ---- NEW TABLE {newTable}")
        oldTable = oldTable.drop(['id_date', 'date'], axis = 1)
        data_cache = oldTable.to_dict('records')[0]
        isDataEqual_checkList = list()
        if len(oldTable) != 0:
            for column in list(oldTable.columns):
                if column in list(newTable.columns):
                    if oldTable[column].values[0] != newTable[column].values[0]:
                        assert type(column) == str
                        data_cache[column] = newTable[column].values[0]
                        isDataEqual_checkList.append(False)
                    else:
                        isDataEqual_checkList.append(True)
        is_diff = not any(isDataEqual_checkList)
        return is_diff, data_cache

    def checkPrimaryKeyExist(self, value, oldTable):
        is_exist = False
        if value in oldTable["id_date"].values:
            is_exist = True
        return is_exist
    
    def checkTableExist(self, targetTableName):
        is_tableExist = False
        if targetTableName in self.tableInDB:
            is_tableExist = True
        return is_tableExist
    
    
# # TESTING FUNCTION
# if __name__ == '__main__':
#     from pandas.io import sql
#     import numpy as np
#     import pandas as pd
#     import os
#     import socket
#     server = 
#     database = 
#     username = 
#     password = 
#     driver = '{ODBC Driver 17 for SQL Server}' # at NS_HomeServer
    
    
#     # INSERT DATA
#     module = dbJob(server, database, username, password, driver) # define module 
#     # print(module.getData_slice(10, "gdphh_rawData"))
#     print(module.tableInDB)
