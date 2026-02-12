<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor version="1.0.0"
    xsi:schemaLocation="http://www.opengis.net/sld StyledLayerDescriptor.xsd"
    xmlns="http://www.opengis.net/sld"
    xmlns:ogc="http://www.opengis.net/ogc"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <NamedLayer>
    <Name>water_polys</Name>
    <UserStyle>
      <Title>Water Polygons Style</Title>
      <Abstract>Blue semi-transparent water bodies with dark outline</Abstract>

      <FeatureTypeStyle>
        <Rule>
          <Title>Water</Title>

          <PolygonSymbolizer>
            <Fill>
              <CssParameter name="fill">#4A90E2</CssParameter>
              <CssParameter name="fill-opacity">0.6</CssParameter>
            </Fill>
            <Stroke>
              <CssParameter name="stroke">#1C3F94</CssParameter>
              <CssParameter name="stroke-width">0.8</CssParameter>
              <CssParameter name="stroke-linejoin">round</CssParameter>
            </Stroke>
          </PolygonSymbolizer>

        </Rule>
      </FeatureTypeStyle>

    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>
