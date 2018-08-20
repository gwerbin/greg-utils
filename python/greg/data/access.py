import collections as co


def pyodbc_connection_string(server, username=None, password=None, database=None, native=None):
    conn_param = co.OrderedDict()

    if native is None:
        if platform.system() == 'Windows':
            native = True
        else:
            native = False

    if native:
        conn_param['Driver']= '{SQL Server Native Client 11.0}'
    else:
        conn_param['Driver'] = '{ODBC Driver 13 for SQL Server}'

    conn_param['Server'] = server

    if database is not None:
        conn_param['Database'] = database

    if username is not None:
        conn_param['UID'] = username

    if password is not None:
        conn_param['PWD'] = password

    return ';'.join('{}={}'.format(k, v) for k, v in conn_param.items())
