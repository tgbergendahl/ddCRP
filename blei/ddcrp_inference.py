import numpy as np

# Returns customers that are connected to i, directly or indirectly
def connections(i, links):
    visited = set()
    to_visit = [i]
    while to_visit:
        curr = to_visit.pop(0)
        visited.add(curr)
        pointers = np.where(np.array(links) == curr)[0]
        for p in pointers:
            if p not in visited:
                to_visit.append(p)
    return list(visited)
