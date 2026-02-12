import requests

GS = "http://localhost:8080/geoserver"
AUTH = ("admin", "geoserver")

ws = "sar101"
store = "gis_postgis"

def post_xml(url, xml):
    r = requests.post(url, auth=AUTH, headers={"Content-Type": "text/xml"}, data=xml.encode("utf-8"))
    # 201 Created is typical; 409 Conflict if it already exists
    if r.status_code not in (200, 201, 409):
        raise RuntimeError(f"{r.status_code} {r.text}")
    return r.status_code

# 1) workspace
post_xml(f"{GS}/rest/workspaces", f"<workspace><name>{ws}</name></workspace>")

# 2) datastore (PostGIS)
datastore_xml = f"""
<dataStore>
  <name>{store}</name>
  <connectionParameters>
    <host>postgis</host>
    <port>5432</port>
    <database>gis</database>
    <user>gis</user>
    <passwd>gis</passwd>
    <dbtype>postgis</dbtype>
  </connectionParameters>
</dataStore>
""".strip()
post_xml(f"{GS}/rest/workspaces/{ws}/datastores", datastore_xml)

# 3) publish layers
for layer in ["water_polys_t1", "change_polys_ratio_db"]:
    post_xml(
        f"{GS}/rest/workspaces/{ws}/datastores/{store}/featuretypes",
        f"<featureType><name>{layer}</name></featureType>",
    )

print("Done.")
