function [Date, DataMatrix, ColumnNames]=GetFromWRDS(username, password, varlist, lib_ds, conditions)

% quert statement is of the form
% "select var1, var2, ... from library.database where CONDITIONS"
% '*' instead of varlis selects all variables
% examples of CONDITIONS are
% 'where date between '1960-01-01' and '1965-01-01' and var1 < 0'

% lib_ds=strcat(' OPTIONM.ZEROCD');
% varlist='date, days, rate';  % use '*' to select all columns
% conditions = 'days <= 180';
 

driver = eval('org.postgresql.Driver');
dbURL = 'jdbc:postgresql://wrds-pgdata.wharton.upenn.edu:9737/wrds?ssl=require&sslfactory=org.postgresql.ssl.NonValidatingFactory';
% username = 'mariagrith';
% password = '7CD0Y&j7IMnr';
WRDS = java.sql.DriverManager.getConnection(dbURL, username, password);

if isempty(conditions)
    statement = ['select ' varlist ' from ' lib_ds];
else
        statement = ['select ' varlist ' from ' lib_ds ' where ' conditions];
end

q = WRDS.prepareStatement(statement);
rs = q.executeQuery();

% Get the column names and data types from the ResultSet's metadata
MetaData = rs.getMetaData;
numCols = MetaData.getColumnCount;
data = cell(0,numCols);  % initialize
for colIdx = numCols : -1 : 1
    ColumnNames{colIdx} = char(MetaData.getColumnLabel(colIdx));
    ColumnType{colIdx}  = char(MetaData.getColumnClassName(colIdx));
end
ColumnType = regexprep(ColumnType,'.*\.','');

% Loop through result set and save data into a MATLAB cell array:
rowIdx = 1;
while rs.next
    for colIdx = 1 : numCols
        switch ColumnType{colIdx}
            case {'Float','Double'}
                data{rowIdx,colIdx} = rs.getDouble(colIdx);
            case {'Long','Integer','Short','BigDecimal'}
                data{rowIdx,colIdx} = double(rs.getDouble(colIdx));
            case 'Boolean'
                data{rowIdx,colIdx} = logical(rs.getBoolean(colIdx));
            otherwise
                data{rowIdx,colIdx} = char(rs.getString(colIdx));
        end
    end
    rowIdx = rowIdx + 1;
end
 

DataMatrix = zeros(rowIdx-1,numCols);
Date = [];
for colIdx = 1 : numCols
    if strcmp(ColumnType{colIdx},'Date') == 1
        Date = [Date data(:,colIdx)];
        DataMatrix(:,colIdx) = datenum(data(:,colIdx));
    elseif strcmp(ColumnType{colIdx},'String') == 1 && strcmp(ColumnNames{colIdx},'cp_flag')
        DataMatrix(:,colIdx) = ones(rowIdx-1,1)-2*strncmp(data(:,colIdx),'P',1);
    else
        DataMatrix(:,colIdx) = cell2mat(data(:,colIdx));
    end
end

% Clean up
rs.close();
WRDS.close();

