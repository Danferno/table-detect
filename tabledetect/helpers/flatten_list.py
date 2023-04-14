def flattenList(nestedList):
    if isinstance(nestedList, list):
        newList = []
        for subList in nestedList:
            if isinstance(subList, list):
                for item in subList:
                    newList.append(item)
            else:
                newList.append(subList)
        return newList
    else:
        return nestedList