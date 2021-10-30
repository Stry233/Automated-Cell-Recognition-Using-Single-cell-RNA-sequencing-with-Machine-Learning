def getParsing(inputRawData,mode):
    """
    This function will return the info based
    on input but exclues all unecessary info
    contain in the file
    
    Parameters
    ----------
    inputRawData : str
        the input file
        
    mode : str
        the mode (train, test, all)
    
    Returns
    -------
    dict
        key : str
            name of cell
        value : set(int)
            all id
    """
    # initialize all variable
    res = dict()
    nowKey = ""
    
    # delete all empty line then convert to list for traverse
    dataSplit = [x for x in inputRawData.split("\n") if x!='']
    
    # traverse start, input info in dict
    for line in dataSplit:
        lineSegment = line.split()
        idParsing = lineSegment[0].split(":")
        if (len(idParsing) == 2 and idParsing[0].isalpha() and idParsing[1].isdigit()):
            nowKey = " ".join(str(i) for i in lineSegment[1::])
        elif (len(lineSegment) == 3 and lineSegment[2].isdigit()):
            if (lineSegment[0][0] == '*' and (mode == "test" or mode == "all")): # case: test
                if (not nowKey in res.keys()):
                    res[nowKey] = set()
                res[nowKey].add(int(lineSegment[0][1::]))
            elif (lineSegment[0].isdigit() and (mode == "train" or mode == "all")): # case: train
                if (not nowKey in res.keys()):
                    res[nowKey] = set()
                res[nowKey].add(int(lineSegment[0]))
    return res  


def getId(inputFile):
    """
    This function will parsing the H5
    file and get all ID necessary
    
    Parameters
    ----------
    inputFile: pandas.io.pytables.HDFStore
        the H5 dataset read previously
    
    Returns
    -------
    list[int]:
        all id
    """      
    res = []
    rawData = inputFile['accessions'].keys()
    for element in rawData:
        res.append(int(element.split("_")[0]))
    return res

def getType(inputId, inputDict):
    """
    This function will return the type of
    cell based on the id and correstbonding
    dic variable
    
    Parameters
    ----------
    inputId : int
        cell identification number
        
    inputDict: dict
         The dic of cell type
    
    Returns
    -------
    string
        cell name, return "N/A(#inputId)" when unable to find
    """      
    for (key,values) in inputDict.items():
        for value in values:
            if (value == inputId):
                return key
    fail = "N/A " + str(inputId)
    return fail