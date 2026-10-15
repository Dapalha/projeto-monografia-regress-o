"""Interface web (Gradio) para classificar o arquivo de um rastreador."""
import tempfile
from pathlib import Path

import gradio as gr

from .analise import analisar_arquivo, grafico_risco, tabela_episodios, tabela_resumo


def _cartao(info):
    fraude = info["consenso"] == "FRAUDE"
    cor = "#c62828" if fraude else "#2e7d32"
    return (
        f"<div style='padding:16px;border-radius:10px;border:2px solid {cor};"
        f"background:{cor}14;color:{cor};text-align:center'>"
        f"<div style='font-size:28px;font-weight:700'>{info['consenso']}</div>"
        f"<div style='font-size:15px'>Modelos que indicaram fraude: {info['votos_fraude']}</div></div>"
        f"<p><b>Rastreador:</b> {info['rastreador']} &nbsp;|&nbsp; <b>Período:</b> {info['periodo']}"
        f" &nbsp;|&nbsp; <b>Janelas analisadas:</b> {info['janelas']}</p>"
    )


def criar_interface(pacote):
    nomes = list(pacote["modelos"])

    def executar(caminho, escolha):
        if not caminho:
            return "<p>Envie um arquivo de auditoria.</p>", None, None, None, None
        modelos = nomes if escolha == "Todos" else [escolha]
        try:
            info, resultados, janelas = analisar_arquivo(caminho, pacote, modelos)
        except Exception as e:
            erro = f"<p style='color:#c62828'>Não foi possível analisar o arquivo: {type(e).__name__}: {e}</p>"
            return erro, None, None, None, None

        resumo = tabela_resumo(resultados)
        saida = Path(tempfile.gettempdir()) / f"{Path(caminho).stem}_resultado.csv"
        resumo.assign(arquivo=info["arquivo"], periodo=info["periodo"]).to_csv(saida, sep=";", index=False)
        figura = grafico_risco(janelas, resultados, pacote["decisao"])
        return _cartao(info), figura, resumo, tabela_episodios(resultados), str(saida)

    with gr.Blocks(title="Detecção de fraude em rastreadores") as demo:
        gr.Markdown("## Detecção de fraude em rastreadores veiculares\n"
                    "Envie a planilha de auditoria de **um** rastreador (CSV separado por ';' ou XLSX).")
        with gr.Row():
            arquivo = gr.File(label="Arquivo de auditoria", file_types=[".csv", ".xlsx", ".xls"],
                              type="filepath")
            modelo = gr.Dropdown(["Todos"] + nomes, value="Todos", label="Modelo")
        botao = gr.Button("Analisar", variant="primary")
        cartao = gr.HTML()
        grafico = gr.Plot(label="Risco ao longo do tempo")
        resumo = gr.Dataframe(label="Resultado por modelo")
        episodios = gr.Dataframe(label="Episódios com indícios", wrap=True)
        download = gr.File(label="Baixar resultado (CSV)")
        botao.click(executar, [arquivo, modelo], [cartao, grafico, resumo, episodios, download])
    return demo
