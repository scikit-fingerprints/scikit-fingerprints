To fix benchmark issues under windows modify dataset.py of ogb after line 99 so it looks like this:

```
    if self.meta_info['additional node files'] == 'None':
        additional_node_files = []
    elif type(self.meta_info['additional node files']) is str: #hotfix else -> elif
        additional_node_files = self.meta_info['additional node files'].split(',')
    else:
        additional_node_files = []


    if self.meta_info['additional edge files'] == 'None':
        additional_edge_files = []
    elif type(self.meta_info['additional edge files']) is str: #hotfix else -> elif
        additional_edge_files = self.meta_info['additional edge files'].split(',')
    else:
        additional_edge_files = []
    
    if self.binary:
        self.graphs = read_binary_graph_raw(raw_dir, add_inverse_edge = add_inverse_edge)
    # hotfix extra elif
    elif additional_edge_files == [] and additional_node_files == []:
        self.graphs = None
    else:
        self.graphs = read_csv_graph_raw(raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files)
```