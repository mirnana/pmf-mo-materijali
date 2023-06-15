<?php require_once __SITE_PATH . '/view/_header.php'; ?>

Autor: <?php $author_username ?><br />
Naslov: <?php $project_title ?><br />
Opis: <?php $abstract ?><br />
Ciljani broj članova: <?php $number_of_members ?><br />
Trenutni broj članova: <?php $current_number_of_members ?><br />


<form method="POST" action="<?php echo __SITE_URL; ?>/index.php?rt=login/logout">
    <button type="submit">Odjava</button>
</form>

<?php require_once __SITE_PATH . '/view/_footer.php'; ?>