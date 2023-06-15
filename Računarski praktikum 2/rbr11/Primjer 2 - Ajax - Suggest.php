<?php
$imena = [ "Ana", "Ante", "Boris", "Maja", "Marko", "Mirko", "Slavko", "Slavica" ];
$q = $_GET[ "q" ];

// Protrči kroz sva imena i vrati HTML kod <option> za samo ona 
// koja sadrže string q kao podstring.
foreach( $imena as $ime )
    if( strpos( $ime, $q ) !== false )
        echo "<option value='" . $ime . "' />\n";
?>
