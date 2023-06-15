<?php

//index.php?rt=users/index -> usersController->index()
//index.php?rt=books/search -> booksController->search()
//index.php?rt=con/action -> conController->action()
//index.php?rt=users -> usersController->index()
//index.php -> usersController->index()

if(!isset($_GET['rt'])) {
    $con = 'users';
    $action = 'index';

}
else {
    $route = $_GET['rt'];
    $con_action = explode('/', $route);
    if(count($con_action) == 2) {
        $con = $con_action[0];
        $action = $con_action[1];
    }
    else {
        $con = $con_action[0];
        $action = 'index';
    }
}
// sad je con kontroler, a aktion akcija koju treba pozvari

$conName = $con . 'Controller';

require_once __DIR__ . '/controller/' . $conName . '.php';

$c = new $conName;
$c ->$action();

?>