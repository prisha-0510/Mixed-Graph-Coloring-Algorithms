import gurobipy as gp
import networkx as nx

def calculate_mis_with_gurobi(graph, time_limit=3600):
    """
    Calculate the Maximum Independent Set (MIS) for a given graph using Gurobi solver.

    Parameters:
    - graph (networkx.Graph): The graph for which to compute the MIS.
    - time_limit (int): Time limit in seconds for Gurobi to find the solution.

    Returns:
    - dict: A dictionary where keys are node IDs, and values are binary values (1 for in MIS, 0 for not in MIS).
    """

    options = {
        "WLSACCESSID": "e6e3aa6a-d667-48f2-8648-12da0a56a082",
        "WLSSECRET": "386f3716-8968-4355-9d5a-9514d4e65fa1",
        "LICENSEID": 2588402,
    }
    
    
    with gp.Env(params=options) as env, gp.Model("MIS", env=env) as model:
  
    
        
        model.setParam("TimeLimit", time_limit)  
        model.setParam("OutputFlag", 0)         


        node_vars = {node: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{node}") for node in graph.nodes}
        
        
        model.setObjective(gp.quicksum(node_vars[node] for node in graph.nodes), gp.GRB.MAXIMIZE)
        

        for u, v in graph.edges:
            model.addConstr(node_vars[u] + node_vars[v] <= 1, f"edge_{u}_{v}")
        

        model.optimize()


        d = {node: int(node_vars[node].X > 0.5) for node in graph.nodes}

        return d