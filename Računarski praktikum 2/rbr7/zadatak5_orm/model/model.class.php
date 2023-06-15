<?php 

// Zadatak (srednje-dosta težak.)
// Ovo je samo kostur apstraktne klase Model.
// Trebate sami napisati implementaciju svih funkcija tako da rade kao što je opisano.
// Uputa: trebat ćete koristiti funkcije poput get_called_class(), kao i stvari poput $obj = new $className();
//
// Pogledajte i moguće dodatne funkcije i relacije ovdje:
// https://laravel.com/docs/master/eloquent
// https://laravel.com/docs/master/eloquent-relationships


require_once __DIR__ . '/../app/database/db.class.php';

spl_autoload_register( function ($class_name) 
{
    $fileName = __DIR__ . '/' . strtolower($class_name) . '.class.php';

    if( file_exists( $fileName ) === false )
        return false;

    require_once $fileName;

    return true;
} );


abstract class Model
{
    // Tablica u bazi podataka pridružena modelu. Svaka izvedena klase će definirati svoju.
    protected static $table = null;

    // Asocijativno polje $columns:
    // - ključevi = imena stupaca u bazi podataka u tablici $table;
    // - svakom ključu je pridružena vrijednost koja u bazi piše za objekt $this (onaj čiji je id jedak $this->id).
    protected $columns = [];

    public function __get( $col )
    {
        // Omogućava da umjesto $this->columns['name'] pišemo $this->name.
        // (uoči: $this->columns može ostati protected!)
        if( isset( $this->columns[ $col ] ) )
            return $this->columns[ $col ];

        return null;
    }

    public function __set( $col, $value )
    {
        // Omogućava da umjesto $this->columns['name']='Mirko' pišemo $this->name='Mirko'.
        // (uoči: $this->columns može ostati protected!)
        $this->columns[$col] = $value;

        return $this;
    }

    public static function all()
    {
        // TODO:
        // Funkcija vraća polje koje sadrži sve objekte iz tablice $table.
    }

    public static function find( $id )
    {
        // TODO:
        // Funkcija vraća onaj (jedini!) objekt iz tablice $table kojem je id jednak $id.
    }

    public static function where( $column, $value )
    {
        // TODO:
        // Funkcija vraća polje koje sadrži sve objekte iz tablice $table kojima u stupcu $column piše vrijednost $value.
    }

    public function belongsTo( $className, $foreign_key )
    {
        // TODO
        // Objekt $this ima svojstvo $foreign_key koje predstavlja strani ključ. Taj strani ključ je id od objekta klase
        // $className. Funkcija vraća taj objekt (tipa $className).
    }

    public function hasMany( $className, $foreign_key )
    {
        // TODO
        // Objekt $this ima puno objekata tipa $className.
        // U tablici od $className postoji stupac s imenom $foreign_key sadrži id-ove objekata istog tipa kao što je $this.
        // Objekti čiji je $foreign_key jednak $this->id su oni koji pripadaju $this-u.
        // Funkcija vraća polje tih objekata (tipa $className).
    }    

    public function hasOne( $className, $foreign_key )
    {
        // TODO
        // Kao hasMany, ali postoji samo jedan takav objekt.
        // Funkcija vraća taj jedan objekt (a ne polje).
    }

    public function save()
    {
        // TODO
        // Funkcija sprema novi ili ažurira postojeći redak u tablici $table koji pripada objektu $this.
        // ($this->id je ključ u tablici $table).
    }
}

?>
