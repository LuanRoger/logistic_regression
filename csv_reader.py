from typing import Optional

def read_csv(file_path, skip_header, ignore_column: Optional[list] = None, delimiter=",") -> list:
    csv_data = []
    with open(file_path, "r") as f:
        if skip_header:
            next(f)
        for line in f:
            line_info = _handle_line(line, delimiter, ignore_column)
            if(line_info == None):
                continue

            csv_data.append(line_info)
    
    return csv_data

def _handle_line(line: str, delimiter: str, ignore_column) -> str | None:
    line = line.strip()
    if(line == ""):
        return None
    line_info = line.split(delimiter)
    for ignored_column_index in ignore_column:
        line_info.pop(ignored_column_index)

    return line_info