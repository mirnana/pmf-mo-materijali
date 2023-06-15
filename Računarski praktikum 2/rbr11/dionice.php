<?php
require_once './db.class.php';

function sendJSONandExit( $message ) {
    // Kao izlaz skripte pošalji $message u JSON formatu i
    // prekini izvođenje.
    header( 'Content-type:application/json;charset=utf-8' );
    echo json_encode( $message );
    flush();
    exit( 0 );
}

$timestamp = $_GET['timestamp'];
$last_modifies = 0;

while($last_modifies <=$timestamp) {
    $db = DB::getConnection();
    $st = $db->prepare("SELECT MAX(updated_at) AS last_update FROM Dionice");
    $st->execute();

    $row = $st->fetch();
    $last_modifies = $row['last_update'];

    usleep(1000000);
}

$msg = [];
$msg['timestamp'] = $last_modifies;
$msg['info']=[];

$db = DB::getConnection();
    $st = $db->prepare("SELECT * FROM Dionice");
    $st->execute();


    while($row = $st->fetch()) {
        $dionica = [];
        $dionica['oznaka']=$row['oznaka'];
        $dionica['ime']=$row['ime'];
        $dionica['cijena']=$row['cijena'];

        $msg['info'][]=$dionica;
    }