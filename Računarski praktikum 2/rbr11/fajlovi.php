<?php

function sendJSONandExit( $message ) {
    // Kao izlaz skripte pošalji $message u JSON formatu i
    // prekini izvođenje.
    header( 'Content-type:application/json;charset=utf-8' );
    echo json_encode( $message );
    flush();
    exit( 0 );
}

$m = [];
if(!isset($_GET['fileName'])) {
    $m["greska"]="Nije poslčao ime datoteke";
    sendJSONandExit($m);
}

$fileName = $_GET['fileName'];
$fileNameBase = basename($fileName);

$filesize = filesize($fileNameBase);
$filemtime = filemtime($fileNameBase);

$m["fileName"] = $fileNameBase;
$m["filesize"] = $filesize;
$m['filemtime'] = $filemtime;
?>