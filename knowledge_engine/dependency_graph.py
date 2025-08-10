"""
Dependency Graph Builder Module

Constructs function dependency graphs from AST parsing results and provides
topological sorting capabilities for intelligent summary generation.
"""

import networkx as nx
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Any
from loguru import logger

from code_parser.services.graph_service import MemgraphIngestor


class DependencyGraphBuilder:
    """Builds and analyzes function dependency graphs for topological processing."""
    
    def __init__(self, ingestor: MemgraphIngestor, project_name: str):
        self.ingestor = ingestor
        self.project_name = project_name
        self.call_graph: nx.DiGraph = nx.DiGraph()
        self.function_data: Dict[str, Dict[str, Any]] = {}
        
    def build_call_graph(self) -> nx.DiGraph:
        """
        Build a directed graph representing function call relationships.
        
        Edge direction: callee -> caller (dependency direction)
        If function A calls function B, we create edge B -> A
        This represents that A depends on B.
        """
        logger.info("Building function call dependency graph...")
        
        # Query all functions/methods in the project
        functions_query = """
        MATCH (f)
        WHERE f:Function OR f:Method
        AND f.qualified_name STARTS WITH $project_name
        RETURN f.qualified_name as qn, 
               f.name as name,
               f.file_path as file_path,
               f.start_line as start_line,
               f.end_line as end_line,
               f.source_code as source_code,
               f.docstring as docstring,
               labels(f) as labels
        """
        
        functions = self.ingestor.fetch_all(functions_query, {"project_name": self.project_name})
        logger.info(f"Found {len(functions)} functions/methods")
        
        # Store function metadata
        for func in functions:
            qn = func['qn']
            self.function_data[qn] = func
            self.call_graph.add_node(qn, **func)
            
        # Query call relationships (only within project)
        calls_query = """
        MATCH (caller)-[r:CALLS]->(callee)
        WHERE (caller:Function OR caller:Method) 
        AND (callee:Function OR callee:Method)
        AND caller.qualified_name STARTS WITH $project_name
        AND callee.qualified_name STARTS WITH $project_name
        RETURN caller.qualified_name as caller_qn,
               callee.qualified_name as callee_qn,
               r.call_count as call_count,
               r.line_numbers as line_numbers
        """
        
        calls = self.ingestor.fetch_all(calls_query, {"project_name": self.project_name})
        logger.info(f"Found {len(calls)} internal function calls")
        
        # Add edges: callee -> caller (dependency direction)
        for call in calls:
            caller_qn = call['caller_qn']
            callee_qn = call['callee_qn']
            
            # Skip self-calls for now (recursion handling)
            if caller_qn == callee_qn:
                continue
                
            # Add edge: callee -> caller (caller depends on callee)
            if callee_qn in self.call_graph and caller_qn in self.call_graph:
                self.call_graph.add_edge(
                    callee_qn, 
                    caller_qn,
                    call_count=call.get('call_count', 1),
                    line_numbers=call.get('line_numbers', [])
                )
        
        logger.info(f"Built dependency graph with {self.call_graph.number_of_nodes()} nodes "
                   f"and {self.call_graph.number_of_edges()} edges")
        
        return self.call_graph
    
    def detect_strongly_connected_components(self) -> List[List[str]]:
        """
        Detect strongly connected components (SCCs) in the call graph.
        Each SCC represents a group of functions with circular dependencies.
        """
        if not self.call_graph:
            self.build_call_graph()
            
        sccs = list(nx.strongly_connected_components(self.call_graph))
        
        # Filter out single-node SCCs (no circular dependencies)
        circular_sccs = [list(scc) for scc in sccs if len(scc) > 1]
        
        if circular_sccs:
            logger.warning(f"Found {len(circular_sccs)} strongly connected components "
                          f"(circular dependencies)")
            for i, scc in enumerate(circular_sccs):
                logger.warning(f"  SCC {i+1}: {scc}")
        else:
            logger.info("No circular dependencies detected")
            
        return circular_sccs
    
    def get_topological_batches(self) -> List[List[str]]:
        """
        Perform topological sorting and group functions into batches.
        Functions in the same batch have no dependencies on each other
        and can be processed in parallel.
        
        Returns:
            List of batches, where each batch is a list of function qualified names
        """
        if not self.call_graph:
            self.build_call_graph()
            
        # Handle SCCs first
        sccs = self.detect_strongly_connected_components()
        
        # Create a condensed graph where each SCC is a single node
        condensed_graph = nx.condensation(self.call_graph)
        
        # Map condensed nodes back to original function names
        scc_mapping = {}  # condensed_node_id -> list of function names
        for node_id in condensed_graph.nodes():
            members = condensed_graph.nodes[node_id]['members']
            scc_mapping[node_id] = list(members)
        
        # Perform topological sort on condensed graph
        try:
            topo_order = list(nx.topological_sort(condensed_graph))
        except nx.NetworkXError as e:
            logger.error(f"Failed to perform topological sort: {e}")
            # Fallback: return all functions as single batch
            return [list(self.function_data.keys())]
        
        # Group by "levels" - functions that can be processed together
        batches = []
        in_degree = {node: condensed_graph.in_degree(node) for node in condensed_graph.nodes()}
        
        while topo_order:
            # Find all nodes with in_degree 0 in current iteration
            current_batch_nodes = [node for node in topo_order if in_degree[node] == 0]
            
            if not current_batch_nodes:
                # This shouldn't happen with a DAG, but handle gracefully
                logger.warning("No nodes with in_degree 0 found, breaking")
                break
            
            # Convert condensed nodes back to function names
            current_batch = []
            for node in current_batch_nodes:
                current_batch.extend(scc_mapping[node])
            
            batches.append(current_batch)
            
            # Remove processed nodes and update in_degrees
            for node in current_batch_nodes:
                topo_order.remove(node)
                for successor in condensed_graph.successors(node):
                    in_degree[successor] -= 1
        
        logger.info(f"Created {len(batches)} processing batches")
        for i, batch in enumerate(batches):
            logger.info(f"  Batch {i+1}: {len(batch)} functions")
            
        return batches
    
    def get_function_dependencies(self, function_qn: str) -> Set[str]:
        """Get all functions that the given function depends on (calls)."""
        if function_qn not in self.call_graph:
            return set()
            
        # Get all predecessors (functions this function calls)
        return set(self.call_graph.predecessors(function_qn))
    
    def get_function_metadata(self, function_qn: str) -> Dict[str, Any]:
        """Get metadata for a specific function."""
        return self.function_data.get(function_qn, {})
    
    def get_batch_context_library(self, batch: List[str]) -> Dict[str, str]:
        """
        Build a context library containing summaries of all dependencies
        needed for the current batch.
        
        Args:
            batch: List of function qualified names in current batch
            
        Returns:
            Dictionary mapping dependency qualified names to their summaries
        """
        context_library = {}
        
        for func_qn in batch:
            dependencies = self.get_function_dependencies(func_qn)
            for dep_qn in dependencies:
                if dep_qn not in context_library:
                    # TODO: Load summary from storage (implement after summary generation)
                    context_library[dep_qn] = f"Summary for {dep_qn} (placeholder)"
        
        return context_library