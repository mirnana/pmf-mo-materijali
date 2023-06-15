<?php
    echo("bleh");
    $message = [];
    if(!isset($_GET['op1']) || !isset($_GET['op2']) || !isset($_GET['op'])) {
        $message["greška"] = "Nisu poslani operatori!";
        sendJSONandExit($message);
    }

    $op1 = $_GET['op1'];
    $op2 = $_GET['op2'];
    $operator = $_GET['op'];

    function sendJSONandExit( $message ) {
        // Kao izlaz skripte pošalji $message u JSON formatu i
        // prekini izvođenje.
        header( 'Content-type:application/json;charset=utf-8' );
        echo json_encode( $message );
        flush();
        exit( 0 );
    }

    if($op === "+") $message['rez'] = $op1 + $op2;
    elseif ($op === "-") $message['rez'] = $op1 - $op2;
    elseif ($op === "*") $message['rez'] = $op1 * $op2;
    elseif ($op === "/") {
        if($op2 === 0) {
            $message["greška"] = "Dijeljenje s nulom!";
            sendJSONandExit($message);        
        }
        $message['rez'] = $op1 / $op2;
    }
    sendJSONandExit($message);
?>