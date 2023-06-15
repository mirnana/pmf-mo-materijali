onmessage = function( e )
{
    var pixelRow = e.data.pixelRow;

    for( var yi = e.data.yStart; yi < e.data.yEnd; ++yi )
    {
        var y = 4*(yi/e.data.ySize) - 2; // y je u [-2, 2]
    
        var numIter;
        for( var xi = 0; xi < e.data.xSize; ++xi )
        {
            var x = 4*(xi/e.data.xSize) - 2; // x je u [-2, 2]
            var xx = x, yy = y;
    
            numIter = 0;
            while( numIter < 255 && xx*xx + yy*yy < 4 )
            {
                // z = xx + i*yy --> z = z^2 + (-0.72 + i*0.18)
                var tmp = xx*xx - yy*yy - 0.7268953;
                yy = 2*xx*yy + 0.188887;
                xx = tmp;
                ++numIter;
            }
        
            pixelRow.data[4*e.data.xSize*(yi-e.data.yStart) + 4*xi] 
                = pixelRow.data[4*e.data.xSize*(yi-e.data.yStart) + 4*xi+1] 
                = pixelRow.data[4*e.data.xSize*(yi-e.data.yStart) + 4*xi+2] 
                = 255 - numIter;
            pixelRow.data[4*e.data.xSize*(yi-e.data.yStart) + 4*xi+3] = 255;
        }
    }

    postMessage( {yStart: e.data.yStart, pixelRow: pixelRow } );
}
