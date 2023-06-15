<?php 

// https://rp2.studenti.math.hr/~username/library/index.php -> users/index
// https://rp2.studenti.math.hr/~username/library/index.php?rt=books -> books/index
// https://rp2.studenti.math.hr/~username/library/index.php?rt=users -> users/index
// https://rp2.studenti.math.hr/~username/library/index.php?rt=users/index
// https://rp2.studenti.math.hr/~username/library/index.php?rt=books/search
// https://rp2.studenti.math.hr/~username/library/index.php?rt=con/action
// https://rp2.studenti.math.hr/~username/library/index.php?rt=

if( !isset( $_GET['rt'] ) )
{
    $controller = 'users';
    $action = 'index';
}
else
{
    $parts = explode( '/', $_GET['rt'] );

    if( isset( $parts[0] ) && preg_match( '/^[A-Za-z0-9]+$/', $parts[0] ) )
        $controller = $parts[0];
    else 
        $controller = 'users';

    if( isset( $parts[1] ) && preg_match( '/^[A-Za-z0-9]+$/', $parts[1] ) )
        $action = $parts[1];
    else 
        $action = 'index';
}

$controllerName = $controller . 'Controller';

if( !file_exists( __DIR__ . '/controller/' . $controllerName . '.php' ) )
    error_404();

require_once __DIR__ . '/controller/' . $controllerName . '.php';

if( !class_exists( $controllerName ) )
    error_404();

$con = new $controllerName();

if( !method_exists( $con, $action ) )
    error_404();

$con->$action();
exit(0);


// ------------------------------------
function error_404()
{
    require_once __DIR__ . '/controller/_404Controller.php';
    $con = new _404Controller();
    $con->index();
    exit(0);
}

?>

