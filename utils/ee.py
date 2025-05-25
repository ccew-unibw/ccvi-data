from utils.gee import GEEClient
import ee


def fc_to_dict(fc: ee.FeatureCollection) -> ee.Dictionary:
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()), selectors=prop_names
    ).get("list")
    return ee.Dictionary.fromLists(prop_names, prop_lists)


def get_grid_chunks(client: GEEClient, chunks: int = 10) -> list:
    """Return a list of tuples with lower and upper bounds of roughly equally sized chunks."""
    pgids = client.asset.aggregate_array("pgid").getInfo()
    pgids.sort()
    chunk_size = len(pgids) // chunks
    result = []
    for i in range(chunks):
        lower = i * chunk_size
        upper = (i + 1) * chunk_size
        if i == chunks - 1:
            chunked_data = pgids[lower:]
            result.append((chunked_data[0], chunked_data[-1]))
        else:
            chunked_data = pgids[lower:upper]
            result.append((chunked_data[0], chunked_data[-1]))
    return result
